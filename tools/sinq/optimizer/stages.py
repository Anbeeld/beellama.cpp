"""Typed stage orchestration. Scoring mathematics live in native/backend modules."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import contextmanager, ExitStack

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'gguf-py'))
from .schemas import load_config, validate_pairing, read_json, SCORERS
from .resources import atomic_write, canonical_json, digest, file_digest, reject_alias, observe_phase
from .cache import CandidateStore
from .generate import generate, preflight
from . import native_codec
from .provenance import candidate_for,graph_assignment,observations


def _json(path):
    return read_json(path)


@contextmanager
def _run(config, *, verified_inputs=None):
    fingerprint=verified_inputs.fingerprint if verified_inputs is not None else file_digest
    path=Path(config['candidate_manifest']).resolve();manifest_sha256=fingerprint(path);run=_json(path)
    if run.get('schema_version')!=2 or run.get('checksum')!=digest({k:v for k,v in run.items() if k!='checksum'}):
        raise ValueError('candidate run manifest schema/checksum mismatch')
    if 'generation_identity_sha256' in run:
        identity=_json(path.parent/'run-identity.json')
        if identity.get('kind')!='sinq-generation-run' or identity.get('checksum')!=run['generation_identity_sha256'] or identity['checksum']!=digest({k:v for k,v in identity.items() if k!='checksum'}):
            raise ValueError('candidate generation identity closure mismatch')
    source=(path.parent/run['source_gguf']).resolve()
    if fingerprint(source)!=run['source_sha256']:
        raise ValueError('candidate source GGUF checksum mismatch')
    cache=(path.parent/run['cache']).resolve()
    if cache!=path.parent/'cache':
        raise ValueError('candidate cache must be the run-owned cache directory')
    b=config['budgets'];store=CandidateStore(cache,max_bytes=b['max_output_bytes'],max_files=b['max_files'])
    with store.verified_session():
        ids=[];requested_scores=0
        for name,record in sorted(run['tensors'].items()):
            for key in [record['stock'],*[r['candidate_id'] for r in record['candidates']]]:
                candidate=candidate_for(run,store,name,key)
                requested_scores+=1
                ids.append(key)
        if len(run['tensors'])>b['max_tensors'] or len(ids)>b['max_candidates']*b['max_tensors']:
            raise ValueError('candidate manifest count budget exceeded')
        if 'assignments' in config:
            requested_scores=sum(len(a) for a in config['assignments'].values()) if config.get('method')=='local' else len(config['assignments'])
        if config['operation']=='score' and requested_scores>b['max_evaluations']:
            raise ValueError('requested candidate scoring work budget exceeded')
        yield run,store,source
        if fingerprint(source)!=run['source_sha256'] or fingerprint(path)!=manifest_sha256:
            raise ValueError('candidate source/manifest changed during stage')


def _validator(binary, *, threads=None):
    if not binary or not Path(binary).is_file():
        raise ValueError('a local native_loader (llama-sinq-model) is required')
    threads=native_codec._thread_count(threads)
    def validate(path):
        result=subprocess.run([str(Path(binary).resolve()),'--load-only',str(Path(path).resolve())],
            env={**os.environ,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads)},
            capture_output=True,timeout=120)
        if result.returncode:
            raise RuntimeError('native loader rejected hybrid; exit '+str(result.returncode))
    return validate


def _stock_baseline_path(config,run):
    recorded=run.get('stock_baseline')
    path=config.get('stock_baseline_manifest')
    if recorded:
        recorded_path=(Path(config['candidate_manifest']).resolve().parent/recorded['manifest']).resolve()
        if file_digest(recorded_path)!=recorded['manifest_sha256']:
            raise ValueError('run stock baseline manifest checksum mismatch')
        if path is not None and Path(path).resolve()!=recorded_path:
            raise ValueError('configured stock baseline differs from candidate run')
        path=recorded_path
    return Path(path).resolve() if path is not None else None


def _stock_baseline(config,run,source,*,prepared=None):
    recorded=run.get('stock_baseline')
    path=_stock_baseline_path(config,run)
    if path is None:return None,None
    if prepared is not None:
        prepared.check(source,path)
        manifest=prepared.manifest
    else:
        from .baseline import load_baseline
        manifest,baseline=load_baseline(path,source,threads=config['threads'])
    if manifest['recipe_sha256']!=run['recipe_sha256'] or manifest['imatrix_sha256']!=run['imatrix_sha256']:
        raise ValueError('candidate run recipe/imatrix differs from complete stock baseline')
    if recorded and (manifest['baseline_sha256']!=recorded['baseline_sha256'] or manifest['recipe_sha256']!=recorded['recipe_sha256']):
        raise ValueError('candidate run stock baseline model/recipe checksum mismatch')
    return Path(path).resolve(),manifest


def assemble_config(config):
    if os.name=='nt':
        from .prepared import VerifiedInputs,PreparedAssembly
        with ExitStack() as session:
            verified=session.enter_context(VerifiedInputs())
            with _run(config,verified_inputs=verified) as inputs:
                run,store,source=inputs
                stock=_stock_baseline_path(config,run)
                prepared=session.enter_context(PreparedAssembly(source,stock,threads=config['threads'],verified_inputs=verified)) if stock else None
                return _assemble(config,*inputs,prepared=prepared,verified_inputs=verified)
    with _run(config) as inputs:return _assemble(config,*inputs)


def _assemble(config,run,store,source,*,prepared=None,verified_inputs=None):
    from .assembly import assemble
    fingerprint=verified_inputs.fingerprint if verified_inputs is not None else file_digest
    if fingerprint(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('assembly source mismatch')
    for name,record in run['tensors'].items():
        if record.get('locked') and config['assignment'].get(name)!=record['locked']:
            raise ValueError('assembly must preserve every locked generation assignment: '+name)
    for name,key in config['assignment'].items():candidate_for(run,store,name,key)
    baseline,locked=_stock_baseline(config,run,source,prepared=prepared)
    return dict(status='PASS',source_sha256=run['source_sha256'],stock_baseline_sha256=locked['baseline_sha256'] if locked else None,
        output_sha256=assemble(source,config['output_gguf'],store,config['assignment'],stock_baseline=baseline,
        native_validate=_validator(config.get('native_loader'),threads=config['threads']),max_output_bytes=config['budgets']['max_output_bytes'],threads=config['threads'],prepared=prepared))


def _new_output(config,inputs):
    output=Path(config['output_directory']).resolve()
    reject_alias(output,inputs)
    if output.exists():raise FileExistsError('immutable optimizer output already exists')
    output.mkdir(parents=True,exist_ok=False)
    return output


def _advanced_reference(checkpoint,source,budget,threads,*,policy=None,monitor=None):
    from .reference_loader import load_reference
    try:
        return load_reference(checkpoint,source,budget,threads,policy=policy,monitor=monitor)
    except Exception as error:
        import traceback
        message=str(error)
        kind=ValueError if isinstance(error,ValueError) else RuntimeError
        traceback.clear_frames(error.__traceback__)
        raise kind(message) from None


def _score_partition(config,run,capture):
    """Bind observed scorer tokens to the reserved calibration/validation streams."""
    recorded=run.get('corpus_partitions')
    if not recorded:return None
    from .corpora import load_corpora,bind_corpora
    import hashlib
    import struct
    path=(Path(config['candidate_manifest']).resolve().parent/recorded['manifest']).resolve()
    if file_digest(path)!=recorded['file_sha256']:raise ValueError('score corpus lock changed')
    partitions=load_corpora(path)
    if config['method']=='local':
        token_sha=capture.get('sampling',{}).get('token_sha256')
    else:
        tokens=capture.get('tokens')
        if not isinstance(tokens,list) or not tokens or any(not isinstance(row,list) or not row for row in tokens):raise ValueError('invalid captured token rows')
        flat=[v for row in tokens for v in row]
        if len(flat)>config['budgets']['max_tokens'] or any(type(v) is not int or not 0<=v<2**31 for v in flat):raise ValueError('invalid captured token IDs/budget')
        token_sha=hashlib.sha256(struct.pack('<'+'I'*len(flat),*flat)).hexdigest()
    bindings=dict(calibration=dict(manifest_sha256=file_digest(config['capture_manifest']),token_sha256=token_sha),validation=None)
    if config.get('selection_policy')=='held_out':
        validation=read_json(config['validation_manifest'])
        bindings['validation']=dict(manifest_sha256=file_digest(config['validation_manifest']),token_sha256=validation.get('sampling',{}).get('token_sha256'))
    return bind_corpora(partitions,bindings)


_SCORER_CONFIGURATION_FIELDS=('scorer_name','method','objective','threads','context','probes','seed',
    'fisher_positions','reference_dtype','device','cuda_devices','memory_policy')
_SCORE_INPUTS={'candidate_manifest_sha256':'candidate_manifest','training_capture_sha256':'capture_manifest',
    'validation_manifest_sha256':'validation_manifest','pairing_manifest_sha256':'pairing_manifest'}


def _score_input_hashes(config):
    return {key:file_digest(config[path]) if config.get(path) is not None else None
        for key,path in _SCORE_INPUTS.items()}


def _scorer_context(config,run,partition_identity,input_hashes,*,reference=None,execution_policy=None,execution_identity=None):
    from .implementation import implementation_fingerprint,native_runtime_fingerprint
    local=config['method']=='local'
    return dict(schema_version=2,source_sha256=run['source_sha256'],**input_hashes,
        selection_policy=config['selection_policy'],data_partition_identity=partition_identity,
        scorer_configuration={key:config.get(key) for key in _SCORER_CONFIGURATION_FIELDS},
        reference_identity=reference,execution_identity_sha256=execution_identity,execution_policy=execution_policy,
        backend_identity=dict(implementation_sha256=implementation_fingerprint()['checksum'],
            native_runtime=native_runtime_fingerprint({'bridge':config['native_bridge']})),
        precision_identity=dict(decoded_weights='float32',activations='float32',output_accumulation='float64') if local else
            {key:execution_policy[key] for key in ('reference_dtype','execution_dtype','parameter_storage_policy')},
        device_identity=dict(devices=['cpu'],device_identifiers=None) if local else
            {key:execution_policy[key] for key in ('devices','device_identifiers')})


def _validate_score_context(value):
    """Check the serialized context closure, including nested tensor observations."""
    context=value.get('scorer_context')
    fields={'schema_version','source_sha256',*_SCORE_INPUTS,'selection_policy','data_partition_identity',
        'scorer_configuration','reference_identity','execution_identity_sha256','execution_policy',
        'backend_identity','precision_identity','device_identity'}
    if not isinstance(context,dict) or set(context)!=fields or context['schema_version']!=2:
        raise ValueError('score context schema mismatch')
    configuration=context['scorer_configuration']
    if not isinstance(configuration,dict) or set(configuration)!=set(_SCORER_CONFIGURATION_FIELDS):
        raise ValueError('score context scorer configuration mismatch')
    if not isinstance(configuration['scorer_name'],str) or SCORERS.get(configuration['scorer_name'])!=(configuration['method'],configuration['objective']):
        raise ValueError('score context scorer name/method/objective mismatch')
    for key in ('source_sha256',*_SCORE_INPUTS):
        sha=context[key]
        if sha is None and key in ('validation_manifest_sha256','pairing_manifest_sha256'):continue
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('score context invalid content hash: '+key)
    configuration_identity=digest(configuration);context_identity=digest(context)
    if (value.get('context_identity')!=context_identity or value.get('scorer_configuration_identity')!=configuration_identity or
            value.get('candidate_manifest_sha256')!=context['candidate_manifest_sha256'] or
            value.get('data_partition_identity')!=context['data_partition_identity'] or
            value.get('method')!=configuration['method'] or value.get('objective')!=configuration['objective'] or
            context['selection_policy'] not in ('in_sample','held_out') or
            context['backend_identity'] is None or context['precision_identity'] is None or context['device_identity'] is None):
        raise ValueError('score context identity/metadata mismatch')
    backend=context['backend_identity']
    if (not isinstance(backend,dict) or set(backend)!={'implementation_sha256','native_runtime'} or
            not isinstance(backend['implementation_sha256'],str) or len(backend['implementation_sha256'])!=64 or
            not isinstance(backend['native_runtime'],dict) or set(backend['native_runtime'])!={'bridge'} or
            not isinstance(backend['native_runtime']['bridge'],dict) or 'executable' not in backend['native_runtime']['bridge']):
        raise ValueError('score context backend identity schema mismatch')
    if context['selection_policy']=='held_out' and any(context[key] is None for key in ('validation_manifest_sha256','pairing_manifest_sha256')):
        raise ValueError('score context held-out provenance missing')
    if configuration['method']=='local':
        if configuration['device']!='cpu' or any(configuration[key] is not None for key in ('context','probes','seed','fisher_positions','reference_dtype','cuda_devices','memory_policy')):
            raise ValueError('local score context has unsupported execution controls')
        if any(context[key] is not None or value.get(key) is not None for key in ('reference_identity','execution_identity_sha256','execution_policy')):
            raise ValueError('local score context has advanced reference metadata')
        if context['precision_identity']!=dict(decoded_weights='float32',activations='float32',output_accumulation='float64') or context['device_identity']!=dict(devices=['cpu'],device_identifiers=None):
            raise ValueError('local score context precision/device mismatch')
    else:
        policy=context['execution_policy']
        if (not isinstance(policy,dict) or set(policy)!={'reference_dtype','execution_dtype','parameter_storage_policy','devices','memory_policy','deterministic','versions','device_identifiers'} or
                any(context[key] is None or value.get(key)!=context[key]
                for key in ('reference_identity','execution_identity_sha256','execution_policy')) or
                context['precision_identity']!={key:policy.get(key) for key in ('reference_dtype','execution_dtype','parameter_storage_policy')} or
                context['device_identity']!={key:policy.get(key) for key in ('devices','device_identifiers')}):
            raise ValueError('advanced score context execution/reference mismatch')
        reference=context['reference_identity']
        if (not isinstance(reference,dict) or reference.get('source_gguf_sha256')!=context['source_sha256'] or
                reference.get('execution_policy')!=policy or
                context['execution_identity_sha256']!=hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest() or
                context['selection_policy']!='in_sample' or any(context[key] is not None for key in ('validation_manifest_sha256','pairing_manifest_sha256'))):
            raise ValueError('advanced score context reference identity closure mismatch')
    def validate_record(record):
        if not isinstance(record,dict) or any(record.get(key)!=expected for key,expected in (
                ('context_identity',context_identity),('scorer_configuration_identity',configuration_identity),
                ('method',configuration['method']),('objective',configuration['objective']),('scorer_name',configuration['scorer_name']))):
            raise ValueError('score record context mismatch')
        if 'capture_sha256' in record and record['capture_sha256']!=context['training_capture_sha256']:
            raise ValueError('score record capture context mismatch')
        if configuration['method']=='local':
            selected='validation_score' if context['selection_policy']=='held_out' else 'train_score'
            if record.get(selected) is None or record.get('score')!=record[selected]:
                raise ValueError('score record selection policy context mismatch')
        members=record.get('tensor_scores',[])
        if not isinstance(members,list):raise ValueError('score record tensor context must be a list')
        for member in members:validate_record(member)
    if not isinstance(value.get('records'),list):raise ValueError('score context records must be a list')
    for record in value['records']:validate_record(record)


def score_config(config):
    with _run(config) as inputs:return _score(config,*inputs)


def _score(config,run,store,source):
    import gguf
    method=config['method'];b=config['budgets']
    if method!='local' and (config.get('selection_policy','in_sample')!='in_sample' or any(k in config for k in ('validation_manifest','pairing_manifest'))):
        raise ValueError('advanced held_out capture provenance is not supported')
    _stock_baseline(config,run,source)
    if not config.get('native_bridge'):raise ValueError('native_bridge is required for actual candidate decoding/scoring')
    input_hashes=_score_input_hashes(config)
    capture=_json(config['capture_manifest']);records=[]
    partition_identity=_score_partition(config,run,capture)
    if method=='local':
        validated=native_codec.validate_activations(config['native_bridge'],config['capture_manifest'],source,threads=config['threads'])
        validation=None
        if config['selection_policy']=='held_out':
            validation=native_codec.validate_activations(config['native_bridge'],config['validation_manifest'],source,threads=config['threads'])
            validate_pairing(_json(config['pairing_manifest']),capture,_json(config['validation_manifest']))
        requests={name:[t['stock'],*[c['candidate_id'] for c in t['candidates']]] for name,t in sorted(run['tensors'].items())}
        if 'assignments' in config:
            requests={}
            for assignment in config['assignments'].values():
                for name,key in assignment.items():
                    candidate_for(run,store,name,key)
                    requests.setdefault(name,set()).add(key)
        reader=gguf.GGUFReader(source)
        try:
            tensors={t.name:t for t in reader.tensors}
            for name,keys in sorted(requests.items()):
                if name not in validated['tensors']:raise ValueError('missing local capture site: '+name)
                a=validated['tensors'][name];v=validation['tensors'][name] if validation else None
                candidates=[(key,candidate_for(run,store,name,key)) for key in sorted(keys)]
                scores=native_codec.rescore_many(config['native_bridge'],candidates,tensors[name],a,validation=v,
                    objective=config['objective'],workspace_bytes=b['workspace_bytes'],threads=config['threads'])
                for key,score in scores:
                    records.append({**score,'candidate_id':key,'tensor':name,'method':'local','objective':config['objective'],
                        'capture_sha256':input_hashes['training_capture_sha256']})
        finally:reader.data._mmap.close()
        if 'assignments' in config:
            individual={(r['tensor'],r['candidate_id']):r for r in records}
            records=[]
            for label,assignment in sorted(config['assignments'].items()):
                members=[individual[(name,key)] for name,key in sorted(assignment.items())]
                records.append(dict(assignment_name=label,assignment=assignment,method='local',objective=config['objective'],
                    train_score=math.fsum(r['train_score'] for r in members),
                    validation_score=math.fsum(r['validation_score'] for r in members) if validation else None,
                    aggregation='sum_of_tensor_mean_output_losses',tensor_scores=members))
    else:
        import torch
        from .blockwise import score_block,block_delta
        from .capture import capture_block
        from .fisher import qwen_probes
        from .execution_backend import resolve_policy
        from .memory_accounting import ResourceMonitor
        policy=resolve_policy(config)
        print('SINQ execution policy: '+canonical_json(policy.identity()).decode(),file=sys.stderr)
        monitor=ResourceMonitor(policy)
        allowed={'schema_version','source_sha256','tokens','layers','attention_mask','position_ids','baseline'}
        if set(capture)-allowed or not {'schema_version','source_sha256','tokens','layers'}<=set(capture) or capture['schema_version']!=2 or capture['source_sha256']!=run['source_sha256']:
            raise ValueError('invalid versioned block capture specification')
        tokens=torch.tensor(capture['tokens'],dtype=torch.long)
        if tokens.ndim!=2 or tokens.numel()>b['max_tokens']:raise ValueError('block token budget exceeded')
        mask=torch.tensor(capture['attention_mask']) if 'attention_mask' in capture else None
        positions=torch.tensor(capture['position_ids'],dtype=torch.long) if 'position_ids' in capture else None
        layers=capture['layers']
        if not isinstance(layers,list) or not layers or len(set(layers))!=len(layers) or any(type(layer)!=int or layer<0 for layer in layers):
            raise ValueError('capture layers must be unique nonnegative integers')
        if len(layers)>b['max_tensors']:raise ValueError('capture layer budget exceeded')
        upstream=graph_assignment(run,store,capture.get('baseline',{}),before_layer=min(layers))
        requested=observations(run,store,config,layers)
        # Check the entire decoded baseline before loading or decoding tensors.
        baseline_bytes=sum(c.provenance['shape'][0]*c.provenance['shape'][1]*
            (12+2*torch.empty((),dtype=policy.dtype).element_size()) for c in upstream.values())
        if baseline_bytes>b['workspace_bytes']:
            raise ValueError('aggregate baseline decode/swap exceeds workspace budget')
        phase_records=2*(2+len(layers)*(1+(method=='fisher_vjp_mc'))+len(requested))+1
        if phase_records*4096>min(b['max_output_bytes'],b['workspace_bytes']):
            raise ValueError('phase diagnostics exceed bounded output/workspace budget')
        monitor.max_records=phase_records
        with monitor.phase('load-and-source-verify'):
            adapter=_advanced_reference(config['checkpoint'],source,b['workspace_bytes'],config['threads'],policy=policy,monitor=monitor)
        if adapter.identity['source_gguf_sha256']!=run['source_sha256']:
            raise ValueError('reference source changed after candidate provenance validation')
        # Cooperative guards at layer boundaries complement polling and allocator caps.
        guard_handles=[layer.register_forward_pre_hook(lambda *args: monitor.check() and None)
                       for layer in adapter.text.layers]
        try:
            with monitor.phase('baseline-decode'):
                baseline={name:native_codec.decode(config['native_bridge'],candidate,workspace_bytes=b['workspace_bytes'],threads=config['threads'])
                          for name,candidate in upstream.items()}
            for layer in layers:
                with monitor.phase(f'capture-{layer}'):
                    locked=capture_block(adapter,tokens,layer,context=config['context'],baseline=baseline,
                        attention_mask=mask,position_ids=positions,max_tokens=b['max_tokens'],max_capture_bytes=b['workspace_bytes'])
                probes=None
                if method=='fisher_vjp_mc':
                    with monitor.phase(f'fisher-{layer}'):
                        fisher_options={}
                        if 'fisher_positions' in config:
                            fisher_options['positions']=config['fisher_positions']
                        probes=qwen_probes(adapter,locked,probes=config['probes'],seed=config['seed'],
                            max_vjps=b['max_probes'],max_cache_bytes=b['workspace_bytes'],**fisher_options)
                for label,index,candidates in requested:
                    if index!=layer:continue
                    if sum(c.provenance['shape'][0]*c.provenance['shape'][1]*(12+2*torch.empty((),dtype=policy.dtype).element_size()) for c in candidates.values())>b['workspace_bytes']:
                        raise ValueError('joint candidate decode/swap workspace budget exceeded')
                    with monitor.phase(f'candidate-{layer}-{label}'):
                        effective={name:native_codec.decode(config['native_bridge'],c,workspace_bytes=b['workspace_bytes'],threads=config['threads']) for name,c in candidates.items()}
                        if probes is None:
                            score=score_block(adapter,locked,effective);value=score[config['objective']]
                        else:
                            value=probes.score(block_delta(adapter,locked,effective));score=dict(provenance=probes.metadata)
                        record=dict(assignment_name=label,assignment={name:c.candidate_id for name,c in candidates.items()},
                            layer=layer,method=method,objective=config['objective'],score=value,details=score)
                        if len(candidates)==1:
                            name=next(iter(candidates));record.update(tensor=name,candidate_id=candidates[name].candidate_id)
                        records.append(record)
                        del effective
                del probes,locked
            adapter.assert_reference()
            monitor.check()
        except (ValueError,RuntimeError,NotImplementedError) as error:
            import traceback
            diagnostic=canonical_json(dict(code='BACKEND_EXECUTION_FAILED',reason=str(error),
                execution_policy=policy.identity(),resource_measurements=monitor.records)).decode()
            # Do not retain failed autograd graphs in chained callee frames.
            traceback.clear_frames(error.__traceback__)
            raise RuntimeError(diagnostic) from None
        except BaseException as error:
            import traceback
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            for handle in guard_handles:handle.remove()
            guard_handles.clear()
            baseline=None; probes=None; locked=None; effective=None
            if sys.exc_info()[0] is not None: adapter=None
    if not records:raise ValueError('requested score stage produced no candidate observations')
    if method=='local':
        # Manifest bytes bind the declared dataset; revalidate its actual data
        # after native subprocesses too, before any score artifact is published.
        for key,expected in [('capture_manifest',validated),*([('validation_manifest',validation)] if validation else [])]:
            if native_codec.validate_activations(config['native_bridge'],config[key],source,threads=config['threads'])!=expected:
                raise ValueError('score capture data changed during scoring')
    if _score_input_hashes(config)!=input_hashes:raise ValueError('score context inputs changed during scoring')
    scorer_context=_scorer_context(config,run,partition_identity,input_hashes,
        reference=adapter.identity if method!='local' else None,
        execution_policy=policy.identity() if method!='local' else None,
        execution_identity=adapter.source_id if method!='local' else None)
    score_config_identity=digest(scorer_context['scorer_configuration']);context_identity=digest(scorer_context)
    def bind_record(record):
        if 'assignment' not in record:record['assignment']={record['tensor']:record['candidate_id']}
        value=record.get('score',record.get('validation_score') if config.get('selection_policy')=='held_out' else record.get('train_score'))
        if value is None:raise ValueError('scorer did not return requested numeric objective')
        record.update(score=value,scorer_name=config['scorer_name'],scorer_configuration_identity=score_config_identity,
            context_identity=context_identity,direction='minimize',scope='tensor' if len(record['assignment'])==1 else 'group')
        for member in record.get('tensor_scores',[]):bind_record(member)
    for record in records:bind_record(record)
    result=dict(schema_version=2,status='PASS',method=method,objective=config['objective'],records=records,scorer_context=scorer_context,
                candidate_manifest_sha256=file_digest(config['candidate_manifest']),context_identity=context_identity,scorer_configuration_identity=score_config_identity,
                data_partition_identity=partition_identity)
    if method!='local':
        result['reference_identity']=adapter.identity
        result['execution_policy']=policy.identity()
        result['resource_measurements']=monitor.records
        result['loading_estimates']=adapter.loading_estimates
        result['execution_identity_sha256']=adapter.source_id
    _validate_score_context(result)
    result['checksum']=digest(result)
    # Validate serialized size before creating an immutable output directory.
    if len(canonical_json(result))>b['max_output_bytes']:
        raise ValueError('score output byte budget exceeded')
    output=_new_output(config,[source,config['candidate_manifest'],config['capture_manifest']])
    atomic_write(output/'scores.json',canonical_json(result),max_bytes=b['max_output_bytes'])
    return result


def _kl_evaluator(config):
    from .kl import NativeKLEvaluator
    from .schemas import native_kl_execution
    e=_json(config['evaluator'])
    allowed={'schema_version','executable','reference','corpus','native_loader','context','batch','microbatch','max_tokens','timeout'}
    if not allowed<=e.keys() or e.keys()-allowed-{'execution','prepared_executable','reference_policy_sha256'} or e['schema_version']!=2:raise ValueError('invalid native evaluator configuration')
    execution=native_kl_execution(e.get('execution',{}))
    if e['max_tokens']>config['budgets']['max_tokens']:raise ValueError('KL token budget exceeded')
    evaluator=NativeKLEvaluator(e['executable'],context=e['context'],batch=e['batch'],microbatch=e['microbatch'],max_tokens=e['max_tokens'],timeout=e['timeout'],threads=config['threads'],execution=execution)
    reference_policy=_json(str(e['reference'])+'.json').get('policy')
    if reference_policy!=evaluator.policy:
        # An explicit opt-in may reuse the ORIGINAL CUDA reference on another
        # identical physical CUDA device. It cannot change numerical settings,
        # binaries, batching, thread policy or the reference-origin evidence.
        if (not e.get('prepared_executable') or e.get('reference_policy_sha256')!=digest(reference_policy)):
            raise ValueError('native KL reference policy identity mismatch')
        original=dict(reference_policy);actual=dict(evaluator.policy)
        original_gpu=original.pop('gpu',None);actual_gpu=actual.pop('gpu',None)
        original['execution']=dict(original['execution']);actual['execution']=dict(actual['execution'])
        original['execution'].pop('cuda_device',None);actual['execution'].pop('cuda_device',None)
        if (original!=actual or not original_gpu or not actual_gpu or
                original_gpu['name']!=actual_gpu['name'] or original_gpu['driver']!=actual_gpu['driver']):
            raise ValueError('prepared cross-device reference changes numerical/runtime policy')
    return e,evaluator


@observe_phase('selection_stage')
def select_config(config,*,resume=None):
    if config['policy'] in ('groupwise_kl','pipeline'):
        from .schemas import native_kl_execution
        native_kl_execution(_json(config['evaluator']).get('execution',{}))
    e=_json(config['evaluator']) if config['policy'] in ('groupwise_kl','pipeline') else {}
    if e.get('prepared_executable'):
        from .prepared import VerifiedInputs,PreparedAssembly
        with ExitStack() as session:
            verified=session.enter_context(VerifiedInputs())
            with _run(config,verified_inputs=verified) as inputs:
                run,store,source=inputs
                stock=_stock_baseline_path(config,run)
                if stock is None:raise ValueError('prepared selection requires complete stock provenance')
                prepared=session.enter_context(PreparedAssembly(source,stock,threads=config['threads'],verified_inputs=verified))
                return _select(config,*inputs,resume=resume,prepared=prepared,verified_inputs=verified)
    with _run(config) as inputs:return _select(config,*inputs,resume=resume)


def _select_kl(config,run,store,source,*,resume=None,prepared=None,verified_inputs=None):
    from .selector import State,Group,coordinate_descent
    from .assembly import assemble
    from .kl import NativeKLEvaluator
    from .run_state import selection_identity,open_selection,selection_lock
    b=config['budgets']
    fingerprint=verified_inputs.fingerprint if verified_inputs is not None else file_digest
    if fingerprint(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('selection source does not match candidate provenance')
    selection=_json(config['selection_manifest']);e=_json(config['evaluator'])
    if selection.get('split')!='selection' or selection.get('corpus_sha256')!=fingerprint(e['corpus']) or selection.get('reference_sha256')!=fingerprint(e['reference']):
        raise ValueError('selection stream/reference provenance mismatch (final-test prohibited)')
    if selection.get('schema_version')!=2 or _json(str(e['reference'])+'.json').get('source_sha256')!=run['source_sha256']:
        raise ValueError('selection reference is not bound to the candidate source')
    from .corpora import load_corpora,bind_corpora
    from .kl import _header
    if 'corpus_partitions_manifest' not in config or 'corpus_bindings' not in run:
        raise ValueError('selection requires corpus_partitions_manifest and token-bound candidate generation provenance')
    partitions=load_corpora(config['corpus_partitions_manifest'])
    corpus_identity=bind_corpora(partitions,run['corpus_bindings'],_header(e['reference'],e['max_tokens'])['tokens'])
    recorded=run.get('corpus_partitions')
    if recorded and (recorded['file_sha256']!=file_digest(config['corpus_partitions_manifest']) or recorded['identity']!=corpus_identity):
        raise ValueError('selection corpus partitions differ from generation lock')
    if set(config['baseline'])!=set(run['tensors']):raise ValueError('baseline must assign the complete run tensor allocation')
    if e['max_tokens']>b['max_tokens']:raise ValueError('KL token budget exceeded')
    if b['max_evaluations']+8>b['max_files']:raise ValueError('selector file budget cannot hold score/checkpoint closure')
    if source.stat().st_size*2>b['max_output_bytes']:raise ValueError('selector disk budget cannot hold a bounded hybrid evaluation')
    e,evaluator=_kl_evaluator(config)
    stock,locked=_stock_baseline(config,run,source,prepared=prepared)
    validator=_validator(e['native_loader'],threads=config['threads'])
    for name,key in config['baseline'].items():candidate_for(run,store,name,key)
    groups=_selection_groups(config,run,store)
    if config['policy']=='pipeline':
        from .selector import filter_groups
        stages=config['stages']
        for stage in stages[:-1]:
            scores=_score_artifact(stage['scores'],config)
            groups=list(filter_groups(State.create(config['baseline'],{}),groups,scores['records'],
                context_identity=scores['context_identity'],keep=stage['keep'],locks=config['locks']))
    adapter=None
    if e.get('prepared_executable'):
        from .prepared_kl import PreparedGroupwiseEvaluator
        if stock is None:raise ValueError('prepared groupwise evaluator requires a complete locked baseline')
        adapter=PreparedGroupwiseEvaluator(evaluator,e['prepared_executable'],e['corpus'],e['reference'],native_loader=e['native_loader'],
            reference_policy=_json(str(e['reference'])+'.json')['policy'])
    identity=selection_identity(config,e,adapter.policy if adapter else evaluator.policy,stock,verified_inputs=verified_inputs)
    inputs=[source,config['candidate_manifest'],config['selection_manifest'],config['evaluator'],e['corpus'],e['reference']]
    if stock:inputs.append(stock)
    output=open_selection(config,resume,identity,inputs,_json)
    with selection_lock(output),ExitStack() as session:
        completed=None
        if (output/'selection.json').exists():
            completed=read_json(output/'selection.json',max_bytes=128<<20)
            if completed.get('checksum')!=digest({k:v for k,v in completed.items() if k!='checksum'}) or completed.get('run_identity_sha256')!=identity['checksum']:
                raise ValueError('completed selection identity/checksum mismatch')
            if not (output/'best.gguf').is_file() or file_digest(output/'best.gguf')!=completed['best_gguf_sha256']:
                raise ValueError('completed selection best checksum mismatch')
        initial=State.create(config['baseline'],{'run_identity_sha256':identity['checksum']})
        if adapter is not None:
            if prepared is None:raise RuntimeError('prepared selection inputs were not established')
            session.enter_context(adapter)
            attempt=1
            while (output/f'runtime-attempt-{attempt}').exists():attempt+=1
            adapter.start(prepared,store,config['baseline'],output/f'runtime-attempt-{attempt}',max_output_bytes=b['max_output_bytes'],verified_inputs=verified_inputs)
        def evaluate(state):
            if adapter is not None:return adapter.evaluate(state)
            with tempfile.TemporaryDirectory(prefix='.evaluation-',dir=output) as temporary:
                path=Path(temporary)/'hybrid.gguf'
                assemble(source,path,store,dict(state.assignment),stock_baseline=stock,
                    native_validate=validator,max_output_bytes=b['max_output_bytes'],threads=config['threads'])
                return evaluator.evaluate(path,e['corpus'],e['reference'],split=config['selection_split'],cache_directory=output/'scores')
        result=coordinate_descent(initial,groups,evaluate,checkpoint=output/'checkpoint.json',
            max_evaluations=b['max_evaluations'],max_passes=config['max_passes'],tolerance=config['tolerance'],split=config['selection_split'],locks=config.get('locks'),
            progress_path=output/'progress.json',recipe=next((part for part in output.parts if part in ('T340','T425')),None),
            evaluator_policy_identity=digest(adapter.policy if adapter else evaluator.policy))
        if completed is not None:
            if completed['assignment']!=dict(result.state.assignment) or completed['selection_kld']!=result.score:
                raise ValueError('completed result differs from resumed checkpoint')
            return completed
        # A crash after best.gguf publication but before selection.json must also
        # be resumable, without overwriting an unrelated or corrupted artifact.
        with tempfile.TemporaryDirectory(prefix='.finalize-',dir=output) as temporary:
            path=Path(temporary)/'best.gguf'
            best_hash=assemble(source,path,store,dict(result.state.assignment),stock_baseline=stock,
                native_validate=validator,max_output_bytes=b['max_output_bytes'],threads=config['threads'],prepared=prepared)
            if adapter is not None:
                from .prepared_kl import token_metrics
                fast_score=adapter.evaluate(result.state)
                rows=output/'final-canonical-selection.tsv'
                evaluator._run(path,e['corpus'],e['reference'],rows)
                canonical_score,canonical_rows=token_metrics(rows,adapter.header['tokens'],evaluator.context)
                _,prepared_rows=token_metrics(adapter.last_execution['token_output'],adapter.header['tokens'],evaluator.context)
                if fast_score!=result.score or canonical_score!=result.score or canonical_rows!=prepared_rows:
                    raise ValueError('published canonical model differs from selected persistent per-token predictions')
                parity={'status':'EXACT_PER_TOKEN_PARITY','model_sha256':best_hash,
                    'selection_kld':canonical_score,'targets':len(canonical_rows),
                    'canonical_execution':evaluator.last_execution,'prepared_execution':adapter.last_execution}
                atomic_write(output/'final-selection-parity.json',canonical_json(parity),max_bytes=1<<20,immutable=False)
            best=output/'best.gguf'
            if best.exists():
                if file_digest(best)!=best_hash:raise ValueError('existing best model collision on resume')
            else:os.link(path,best)
        summary=dict(schema_version=2,status='PASS',run_identity_sha256=identity['checksum'],
            source_sha256=run['source_sha256'],stock_baseline_sha256=locked['baseline_sha256'] if locked else None,
            corpus_partitions=corpus_identity,best_gguf_sha256=best_hash,assignment=dict(result.state.assignment),selection_kld=result.score,
            evaluations=result.evaluations,budget_exhausted=result.budget_exhausted,trace=list(result.trace),
            search_status=result.progress['status'],group_progress=result.progress,
            warning='Selection objective only; not an untouched-test result or quality conclusion. PASS validates publication, not complete group coverage; inspect search_status and group_progress.')
        summary['checksum']=digest(summary)
        atomic_write(output/'selection.json',canonical_json(summary),max_bytes=b['max_output_bytes'])
        return summary


def validate_config_artifacts(config):
    path=Path(config['artifact_manifest']);run=_json(path)
    surrogate=dict(config,candidate_manifest=str(path))
    with _run(surrogate) as (run,store,source):
        if 'source_gguf' in config and file_digest(config['source_gguf'])!=run['source_sha256']:raise ValueError('validation source mismatch')
        _,locked=_stock_baseline(surrogate,run,source)
        return dict(status='PASS',source_sha256=run['source_sha256'],stock_baseline_sha256=locked['baseline_sha256'] if locked else None,
            tensor_count=len(run['tensors']))



def _score_artifact(path,config):
    # Score matrices can legitimately exceed read_json's conservative generic
    # default because their per-record runtime/resource measurements are retained.
    # Respect the explicit artifact budget on the selection config instead.
    value=read_json(path,max_bytes=config['budgets']['max_output_bytes'])
    if (value.get('schema_version')!=2 or value.get('checksum')!=digest({k:v for k,v in value.items() if k!='checksum'}) or
            value.get('candidate_manifest_sha256')!=file_digest(config['candidate_manifest'])):
        raise ValueError('score artifact version/checksum/candidate context mismatch')
    _validate_score_context(value)
    if value['scorer_context']['source_sha256']!=_json(config['candidate_manifest'])['source_sha256']:
        raise ValueError('score artifact source context mismatch')
    return value


def _selection_groups(config,run,store):
    from .selector import Group
    baseline=config['baseline'];locks=config.setdefault('locks',{})
    if set(baseline)!=set(run['tensors']):raise ValueError('initial assignment must exactly cover candidate run')
    for name,record in run['tensors'].items():
        candidate_for(run,store,name,baseline[name])
        if record.get('locked') is not None:
            if baseline[name]!=record['locked'] or locks.get(name,record['locked'])!=record['locked']:
                raise ValueError('selection changed a generation-locked tensor')
            locks[name]=record['locked']
    alternatives=config.get('alternatives',{});groups=[];seen=set()
    for group,members in sorted(config['groups'].items()):
        if seen.intersection(members) or not set(members)<=baseline.keys():raise ValueError('overlapping/unmatched group')
        seen.update(members);locked=set(members)&locks.keys()
        if locked and locked!=set(members):raise ValueError('split mixed locked/unlocked group')
        if group in alternatives:choices=alternatives[group]
        elif len(members)==1:
            n=members[0];r=run['tensors'][n]
            choices=[{n:r['stock']}]+[{n:c['candidate_id']} for c in r['candidates']]
        elif locked:choices=[]
        else:raise ValueError('joint groups require explicit alternatives; no implicit Cartesian product')
        for choice in choices:
            if set(choice)!=set(members):raise ValueError('alternative must exactly assign group members')
            for name,key in choice.items():candidate_for(run,store,name,key)
        groups.append(Group(group,tuple(members),tuple(choices)))
    if seen!=baseline.keys() or not set(alternatives)<=config['groups'].keys():raise ValueError('group/alternative coverage mismatch')
    return groups


def _select(config,run,store,source,*,resume=None,prepared=None,verified_inputs=None):
    if config['policy'] in ('groupwise_kl','pipeline'):return _select_kl(config,run,store,source,resume=resume,prepared=prepared,verified_inputs=verified_inputs)
    from .selector import State,per_group_score_select
    from .assembly import assemble
    from .run_state import open_run,selection_lock
    from .implementation import implementation_fingerprint,native_runtime_fingerprint
    groups=_selection_groups(config,run,store);initial=State.create(config['baseline'],{'source':run['source_sha256']})
    scores=_score_artifact(config['scores'],config) if config['policy']=='per_group_score' else None
    identity=dict(schema_version=2,kind='sinq-assignment-selection',config_sha256=digest(config),
        candidate_manifest_sha256=file_digest(config['candidate_manifest']),source_sha256=file_digest(source),
        score_sha256=file_digest(config['scores']) if scores is not None else None,
        implementation=implementation_fingerprint(),native_runtime=native_runtime_fingerprint({'loader':config['native_loader']}))
    identity['checksum']=digest(identity)
    output=open_run(config,resume,identity,[source,config['candidate_manifest']],_json)
    stock,locked=_stock_baseline(config,run,source)
    with selection_lock(output):
        state=per_group_score_select(initial,groups,scores['records'],context_identity=scores['context_identity'],locks=config['locks']) if scores else initial
        completed=output/'selection.json'
        if completed.exists():
            result=_json(completed)
            if result.get('checksum')!=digest({k:v for k,v in result.items() if k!='checksum'}) or result['run_identity_sha256']!=identity['checksum'] or result['assignment']!=dict(state.assignment) or file_digest(output/'best.gguf')!=result['best_gguf_sha256']:
                raise ValueError('completed selection closure differs on resume')
            return result
        with tempfile.TemporaryDirectory(prefix='.finalize-',dir=output) as temp:
            model=Path(temp)/'model.gguf'
            fingerprint=assemble(source,model,store,dict(state.assignment),stock_baseline=stock,
                native_validate=_validator(config['native_loader'],threads=config['threads']),
                max_output_bytes=config['budgets']['max_output_bytes'],threads=config['threads'])
            target=output/'best.gguf'
            if target.exists():
                if file_digest(target)!=fingerprint:raise ValueError('partial selection model differs')
            else:os.link(model,target)
        result=dict(schema_version=2,status='PASS',policy=config['policy'],run_identity_sha256=identity['checksum'],
            assignment=dict(state.assignment),best_gguf_sha256=fingerprint,source_sha256=run['source_sha256'],
            warning='Frozen-context surrogate assignment, not a claim of global optimum or untouched-test improvement.')
        result['checksum']=digest(result)
        atomic_write(completed,canonical_json(result),max_bytes=config['budgets']['max_output_bytes'])
        return result
