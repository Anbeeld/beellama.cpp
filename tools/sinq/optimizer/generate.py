"""Generic candidate orchestration. All normalization and encoding is native."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import gguf
import numpy as np
from tools.sinq import export_activations as exporter
from . import native_codec
from .schemas import read_json,validate_pairing,parameters
from .resources import canonical_json as canonical,digest,file_digest,atomic_write

NUMERICS='fp16-source-statistics;fp32-best-preupdate;serialized-f16-diagonal;original-W/t;imatrix*t^2'
ROOT=Path(__file__).resolve().parents[3]


def source_identity():
    paths=[ROOT/'src/llama-sinq.cpp',ROOT/'tools/sinq/native-codec.cpp',ROOT/'ggml/src/ggml.c',ROOT/'ggml/src/ggml-quants.c']
    paths+=sorted((ROOT/'src/sinq').glob('*'))
    return digest({p.relative_to(ROOT).as_posix():file_digest(p) for p in paths if p.is_file()})


def _manifest(config,key):
    p=Path(config[key]);value=read_json(p)
    validated=native_codec.validate_activations(config['native_bridge'],p,config['source_gguf'],threads=config['threads'])
    if validated['source_sha256']!=file_digest(config['source_gguf']):raise ValueError('activation/source identity changed')
    return value,validated


def preflight(config,*,resume=None):
    source=Path(config['source_gguf']).resolve();q=config['native_quantizer'];b=config['budgets']
    if not Path(config['native_bridge']).is_file():raise ValueError('configured native bridge unavailable')
    reader=gguf.GGUFReader(source)
    try:
        if reader.fields['general.architecture'].contents()!='qwen35':raise ValueError('only Qwen3.5 projections are eligible')
        if any(k.startswith(('split.','beellama.qk_sinq.')) for k in reader.fields):raise ValueError('split/transformed source unsupported')
        tensors={t.name:t for t in reader.tensors}
        if len(tensors)!=len(reader.tensors):raise ValueError('duplicate source tensor names')
        recipe=q['recipe']
        if any(n not in tensors for n in recipe):raise ValueError('recipe names unknown source tensor')
        for name,qt in recipe.items():
            if qt not in gguf.GGMLQuantizationType.__members__:raise ValueError('unknown exact qtype: '+qt)
        stock=None
        if 'stock_baseline_manifest' in config:
            from .baseline import load_baseline
            locked,_=load_baseline(config['stock_baseline_manifest'],source,threads=config['threads'])
            imatrix=q.get('imatrix_path')
            if locked['imatrix_sha256']!=(file_digest(imatrix) if imatrix else None):raise ValueError('imatrix differs from stock baseline')
            if any(locked['recipe'].get(n)!=qt for n,qt in recipe.items()):raise ValueError('recipe differs from stock allocation')
            stock=dict(manifest=str(Path(config['stock_baseline_manifest']).resolve()),manifest_sha256=file_digest(config['stock_baseline_manifest']),
                baseline_sha256=locked['baseline_sha256'],recipe_sha256=locked['recipe_sha256'])
        # The same native resolver used by llama-quantize resolves selectors/overlaps.
        inventory=[dict(name=n,rows=int(t.shape[1]) if len(t.shape)==2 else 1,columns=int(t.shape[0]),
            qtype=recipe.get(n,t.tensor_type.name).lower(),source_type=t.tensor_type.name.lower()) for n,t in sorted(tensors.items())]
        manual=config.get('parameters',{})
        # Automatic candidates already contain complete settings. Discover targets
        # with the minimum codec-aligned width, not a hidden width-64 candidate.
        discovery_width=32
        direct=dict(schema_version=2,defaults=manual.get('defaults',dict(generator='sinq',strip_width=discovery_width,iterations=16,strength=1.0)),
            targets=config['targets'],overrides=manual.get('overrides',[]))
        resolved=native_codec.resolve_plan(config['native_bridge'],direct,inventory,threads=config['threads'])
        if isinstance(resolved,dict):resolved=resolved['tensors']
        names=[r['name'] for r in resolved]
        if not names or len(set(names))!=len(names) or len(names)>b['max_tensors']:raise ValueError('resolved target count invalid/exceeds limit')
        activations=None;validation=None;pairing=None
        if 'activation_manifest' in config:
            manifest,activations=_manifest(config,'activation_manifest')
            if 'validation_manifest' in config:
                val,validation=_manifest(config,'validation_manifest');pairing=validate_pairing(read_json(config['pairing_manifest']),manifest,val)
        if ('validation_manifest' in config)!=('pairing_manifest' in config):raise ValueError('validation and pairing must be supplied together')
        output=Path(config['output_directory']).resolve()
        if resume is not None:
            if Path(resume).resolve()!=output or not output.is_dir():raise ValueError('resume directory differs from configured existing output')
        elif output.exists():raise FileExistsError('immutable optimizer output exists; use --resume')
        details=[];total=1<<20;files=8;locks=manual.get('locks',{})
        if not isinstance(locks,dict) or not set(locks)<=set(names):raise ValueError('locks name unselected tensors')
        for item in sorted(resolved,key=lambda x:x['name']):
            name=item['name'];t=tensors[name]
            if name not in recipe:raise ValueError('target missing exact recipe qtype: '+name)
            if len(t.shape)!=2 or t.tensor_type.name not in ('F32','F16','BF16'):raise ValueError('source must be original floating matrix')
            k,n=map(int,t.shape);qt=gguf.GGMLQuantizationType[recipe[name]];block,size=gguf.GGML_QUANT_SIZES[qt]
            if recipe[name] not in __import__('tools.sinq.optimizer.schemas',fromlist=['NATIVE_QTYPES']).NATIVE_QTYPES or k%block or n<2:raise ValueError('unsupported target codec/shape')
            if qt.name in ('IQ1_S','IQ2_XXS','IQ2_XS') and 'imatrix_path' not in q:raise ValueError('codec requires frozen imatrix')
            if t.n_bytes+n*k*4>=b['workspace_bytes']//2:raise ValueError('source tensor exceeds resident workspace')
            configs=list(config['resolved_candidates']);lock=locks.get(name);locked_id=None
            if isinstance(lock,dict):configs=[parameters(lock)]
            elif lock is not None:
                if not isinstance(lock,str):raise ValueError('lock must be complete parameters or immutable candidate ID')
                locked_id=lock
            # Explicit override settings are first-class candidates, never substitutes for requested widths.
            if manual.get('overrides') and item['parameters'] not in configs:configs.append(parameters(item['parameters']))
            applicable=[];skipped=[]
            for spec in configs:
                if k%spec['strip_width']:
                    if isinstance(lock,dict):raise ValueError('locked width does not divide input dimension')
                    skipped.append(dict(parameters=spec,reason='input_columns not divisible by requested strip_width'));continue
                applicable.append(spec)
                if spec['generator']=='awq' and activations is None:raise ValueError('AWQ requires verified activation capture')
            if not applicable and locked_id is None:raise ValueError('no applicable candidates: '+name)
            if len(applicable)+1>b['max_candidates']:raise ValueError('candidate limit exceeded after manual resolution')
            for capture in (activations,validation):
                if capture is None:continue
                a=capture['tensors'].get(name)
                if a is None or (a['input_dim'],a['output_dim'])!=(k,n):raise ValueError('capture shape/site mismatch: '+name)
            count=n*k//block*size;total+=(len(applicable)+1)*(count+k*2+16384);files+=(len(applicable)+1)*4
            details.append(dict(name=name,shape=[n,k],qtype=qt.name,packed_bytes=count,candidates=applicable,skipped=skipped,
                parameters=lock if isinstance(lock,dict) else (item['parameters'] if manual.get('defaults') or manual.get('overrides') else None),
                locked=lock is not None,locked_candidate_id=locked_id))
        if total>b['max_output_bytes'] or files>b['max_files']:raise ValueError('preflight artifact/file budget exceeded')
        return dict(schema_version=2,operation='generate',status='PLAN',source_sha256=file_digest(source),
            resolved_candidates=config['resolved_candidates'],tensor_details=details,expected_output_bytes_upper_bound=total,
            expected_files_upper_bound=files,activation=activations,validation=validation,pairing=pairing,stock_baseline=stock)
    finally:reader.data._mmap.close()


def _importance(path,name,k):
    if path is None:return None
    reader=gguf.GGUFReader(path)
    try:
        tensors={t.name:t for t in reader.tensors}
        if name+'.in_sum2' not in tensors or name+'.counts' not in tensors:raise ValueError('imatrix lacks selected tensor: '+name)
        sums=tensors[name+'.in_sum2'];counts=tensors[name+'.counts']
        if sums.tensor_type!=gguf.GGMLQuantizationType.F32 or counts.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('imatrix requires floating sums/counts')
        a=np.asarray(sums.data,dtype=np.float32).reshape(-1);c=np.asarray(counts.data,dtype=np.float32).reshape(-1)
        if a.size!=k or c.size!=1 or not np.isfinite(a).all() or not np.isfinite(c).all() or c[0]<=0 or (a<0).any():raise ValueError('invalid imatrix statistics')
        return (a/c[0]).astype('<f4')
    finally:reader.data._mmap.close()


def _generate(config,plan,output,store,completed,save,verify,bindings,corpus_identity,identity_sha256):
    from .generation_state import validate_completed
    from .implementation import native_runtime_fingerprint
    reader=gguf.GGUFReader(config['source_gguf']);started=time.monotonic()
    source_sha=plan['source_sha256'];q=config['native_quantizer'];b=config['budgets']
    recipe_sha=plan['stock_baseline']['recipe_sha256'] if plan['stock_baseline'] else digest(q['recipe'])
    imatrix_sha=file_digest(q['imatrix_path']) if 'imatrix_path' in q else None
    producer=dict(binary_sha256=file_digest(config['native_bridge']),source_sha256=source_identity(),
        runtime_sha256=digest(native_runtime_fingerprint({'bridge':config['native_bridge']})))
    run=dict(schema_version=2,source_gguf=os.path.relpath(Path(config['source_gguf']).resolve(),output),source_sha256=source_sha,
        cache='cache',recipe_sha256=recipe_sha,imatrix_sha256=imatrix_sha,producer=producer,config_sha256=digest(config),tensors=dict(completed),
        corpus_bindings=bindings,generation_identity_sha256=identity_sha256,status='PASS',purpose='correctness integration; no quality conclusions')
    if corpus_identity is not None:run['corpus_partitions']=dict(manifest=os.path.relpath(Path(config['corpus_partitions_manifest']).resolve(),output),
        file_sha256=file_digest(config['corpus_partitions_manifest']),identity=corpus_identity)
    if plan['stock_baseline']:run['stock_baseline']={**plan['stock_baseline'],'manifest':os.path.relpath(plan['stock_baseline']['manifest'],output)}
    try:
        tensors={t.name:t for t in reader.tensors}
        for details in plan['tensor_details']:
            name=details['name'];tensor=tensors[name];n,k=details['shape']
            if name in completed:
                validate_completed(store,run,name,tensor,details,config);continue
            with tempfile.TemporaryDirectory(prefix='sinq-generate-') as temp:
                d=Path(temp);weights=d/'source.bin';native_codec.copy_tensor_bytes(tensor,weights,max_bytes=b['workspace_bytes'])
                importance=_importance(q.get('imatrix_path'),name,k)
                if importance is not None:importance.tofile(d/'imatrix.f32')
                base=dict(schema_version=2,weights=str(weights),source_type=tensor.tensor_type.name.lower(),rows=n,columns=k,
                    codec=details['qtype'].lower(),threads=config['threads'],workspace_bytes=b['workspace_bytes'],max_output_bytes=b['max_output_bytes'])
                if importance is not None:base['importance']=str(d/'imatrix.f32')
                common=dict(source_sha256=source_sha,tensor_name=name,shape=[n,k],source_qtype=tensor.tensor_type.name,qtype=details['qtype'],
                    input_sha256=file_digest(weights),recipe_sha256=recipe_sha,imatrix_sha256=imatrix_sha,
                    codec=dict(implementation='ggml',chunk_rows=max(1,(16384+k-1)//k),rows_per_expert=n),producer=producer)
                sd=d/'stock-request';sd.mkdir()
                stock=native_codec.request(config['native_bridge'],dict(base,operation='stock',output=str(d/'stock')),sd)
                stock_id=store.put_files(dict(common,sinq=None,generator=dict(generator='stock'),activation=None),d/'stock'/stock['payload'])
                rd=d/'request';rd.mkdir()
                request=dict(base,operation='generate',output=str(d/'generated'),candidates=details['candidates'],max_candidates=b['max_candidates'])
                if plan['activation'] is not None:
                    a=plan['activation']['tensors'][name];request.update(activations=a['data_file'],samples=a['count'],objective='output_l1')
                if plan['validation'] is not None:
                    v=plan['validation']['tensors'][name];request.update(held_out=True,validation=v['data_file'],validation_samples=v['count'])
                found=native_codec.request(config['native_bridge'],request,rd);records=[]
                if len(found['candidates'])!=len(details['candidates']):raise ValueError('native generator returned incomplete candidate coverage')
                for expected,c in zip(details['candidates'],found['candidates']):
                    spec=c['parameters']
                    if spec!=expected:raise ValueError('native candidate specification/order differs from resolved plan')
                    s={key:spec[key] for key in ('strip_width','iterations','strength')};s.update(algorithm='sinkhorn_log',numerics=NUMERICS)
                    activation=None if spec['generator']=='sinq' else dict(source_sha256=plan['activation']['manifest_sha256'],
                        split_sha256=digest(plan['pairing']['train']) if plan['pairing'] else plan['activation']['manifest_sha256'])
                    key=store.put_files(dict(common,sinq=s,generator=spec,activation=activation),d/'generated'/c['payload'],d/'generated'/c['scales'])
                    records.append(dict(candidate_id=key,parameters=spec,train_score=c.get('train_score'),validation_score=c.get('validation_score')))
                locked_id=details['locked_candidate_id']
                if locked_id is not None:
                    # Existing IDs must already be in this immutable registry and match this exact run binding.
                    locked=store.get(locked_id,expected=common)
                    if locked_id not in {r['candidate_id'] for r in records}|{stock_id}:raise ValueError('locked candidate is not materialized by this resolved run')
                elif details['locked']:locked_id=records[0]['candidate_id']
                run['tensors'][name]=dict(stock=stock_id,candidates=records,selected=locked_id or stock_id,locked=locked_id,
                    skipped=details['skipped'],encode_count=found['encode_count'],workspace_bytes=found.get('workspace_bytes',0),elapsed_seconds=found.get('elapsed_seconds',0))
            save(run['tensors'])
        if file_digest(config['source_gguf'])!=source_sha:raise ValueError('source mutated during generation')
        run['diagnostics']=dict(selected_tensors=len(run['tensors']),candidate_count=sum(len(r['candidates']) for r in run['tensors'].values()),
            skipped_count=sum(len(r['skipped']) for r in run['tensors'].values()),execution_policy=dict(threads=config['threads']),
            elapsed_seconds=time.monotonic()-started,cache=dict(store.index_stats))
        # Wall clock/cache counters live in a separate summary so replay preserves the immutable manifest.
        diagnostics=run.pop('diagnostics');run['checksum']=digest(run);verify(run)
        existing=output/'run-manifest.json'
        if existing.exists():
            if existing.read_bytes()!=canonical(run):raise ValueError('resumed immutable manifest differs')
        else:atomic_write(existing,canonical(run),max_bytes=b['max_output_bytes'])
        summary=output/'generation-summary.json'
        if not summary.exists():atomic_write(summary,canonical(dict(schema_version=2,run_sha256=run['checksum'],**diagnostics)),max_bytes=b['max_output_bytes'])
        return run
    finally:reader.data._mmap.close()


def generate(config,*,resume=None):
    if resume is not None:
        output=Path(config['output_directory']).resolve()
        if Path(resume).resolve()!=output or not output.is_dir():raise ValueError('resume directory must be the existing configured output directory')
    from .generation_state import run_generation
    return run_generation(config,resume,_generate)
