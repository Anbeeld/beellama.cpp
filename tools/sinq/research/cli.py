"""One opt-in config entry point: python -m tools.sinq.research.cli CONFIG [--plan]."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'gguf-py'))
from .schemas import load_config, validate_pairing
from .resources import atomic_write, canonical_json, digest, file_digest, reject_alias
from .cache import CandidateStore
from .generate import generate, preflight
from . import native_codec
from .provenance import candidate_for,graph_assignment,observations


def _json(path):
    if Path(path).stat().st_size>8<<20:
        raise ValueError('manifest byte budget exceeded')
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def _run(config):
    path=Path(config['candidate_manifest']).resolve();run=_json(path)
    if run.get('schema_version')!=1 or run.get('checksum')!=digest({k:v for k,v in run.items() if k!='checksum'}):
        raise ValueError('candidate run manifest schema/checksum mismatch')
    if 'generation_identity_sha256' in run:
        identity=_json(path.parent/'run-identity.json')
        if identity.get('kind')!='sinq-generation-run' or identity.get('checksum')!=run['generation_identity_sha256'] or identity['checksum']!=digest({k:v for k,v in identity.items() if k!='checksum'}):
            raise ValueError('candidate generation identity closure mismatch')
    source=(path.parent/run['source_gguf']).resolve()
    if file_digest(source)!=run['source_sha256']:
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
                if candidate.provenance['generator']['family'] in config.get('families',[]):requested_scores+=1
                ids.append(key)
        if len(run['tensors'])>b['max_tensors'] or len(ids)>b['max_candidates']*b['max_tensors']:
            raise ValueError('candidate manifest count budget exceeded')
        if 'assignments' in config:requested_scores=len(config['assignments'])
        if config['operation']=='score' and requested_scores>b['max_evaluations']:
            raise ValueError('requested candidate scoring work budget exceeded')
        yield run,store,source


def _validator(binary, *, threads=None):
    if not binary or not Path(binary).is_file():
        raise ValueError('a local native_loader (llama-sinq-model) is required')
    threads=native_codec._thread_count(threads)
    def validate(path):
        result=subprocess.run([str(Path(binary).resolve()),'--load-only',str(Path(path).resolve())],
            env={**os.environ,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'CUDA_VISIBLE_DEVICES':'0'},
            capture_output=True,timeout=120)
        if result.returncode:
            raise RuntimeError('native loader rejected hybrid; exit '+str(result.returncode))
    return validate


def _stock_baseline(config,run,source):
    recorded=run.get('stock_baseline')
    path=config.get('stock_baseline_manifest')
    if recorded:
        recorded_path=(Path(config['candidate_manifest']).resolve().parent/recorded['manifest']).resolve()
        if file_digest(recorded_path)!=recorded['manifest_sha256']:
            raise ValueError('run stock baseline manifest checksum mismatch')
        if path is not None and Path(path).resolve()!=recorded_path:
            raise ValueError('configured stock baseline differs from candidate run')
        path=recorded_path
    if path is None:return None,None
    from .baseline import load_baseline
    manifest,baseline=load_baseline(path,source)
    if manifest['recipe_sha256']!=run['recipe_sha256'] or manifest['imatrix_sha256']!=run['imatrix_sha256']:
        raise ValueError('candidate run recipe/imatrix differs from complete stock baseline')
    if recorded and (manifest['baseline_sha256']!=recorded['baseline_sha256'] or manifest['recipe_sha256']!=recorded['recipe_sha256']):
        raise ValueError('candidate run stock baseline model/recipe checksum mismatch')
    return Path(path).resolve(),manifest


def lock_baseline_config(config):
    from .baseline import lock_baseline
    return lock_baseline(config['source_gguf'],config['stock_gguf'],config['recipe'],config['output_manifest'],
        quantizer=config['quantizer'],native_validate=_validator(config['native_loader'],threads=config['threads']),
        imatrix=config.get('imatrix_path'),workspace_bytes=config['budgets']['workspace_bytes'],
        max_output_bytes=config['budgets']['max_output_bytes'],threads=config['threads'],timeout=config['timeout'])


def lock_corpora_config(config):
    from .corpora import lock_corpora
    b=config['budgets']
    return lock_corpora(config['token_sources'],config['partitions'],config['output_manifest'],
        max_tokens=b['max_tokens'],max_output_bytes=b['max_output_bytes'],max_files=b['max_files'])


def assemble_config(config):
    with _run(config) as inputs:return _assemble(config,*inputs)


def _assemble(config,run,store,source):
    from .hybrid import assemble
    if file_digest(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('assembly source mismatch')
    for name,key in config['assignment'].items():candidate_for(run,store,name,key)
    baseline,locked=_stock_baseline(config,run,source)
    return dict(status='PASS',source_sha256=run['source_sha256'],stock_baseline_sha256=locked['baseline_sha256'] if locked else None,
        output_sha256=assemble(source,config['output_gguf'],store,config['assignment'],stock_baseline=baseline,
        native_validate=_validator(config.get('native_loader'),threads=config['threads']),max_output_bytes=config['budgets']['max_output_bytes']))


def _new_output(config,inputs):
    output=Path(config['output_directory']).resolve()
    reject_alias(output,inputs)
    if output.exists():raise FileExistsError('immutable research output already exists')
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


def score_config(config):
    with _run(config) as inputs:return _score(config,*inputs)


def _score(config,run,store,source):
    import gguf
    method=config['method'];b=config['budgets']
    if method!='local' and (config.get('selection_policy','in_sample')!='in_sample' or any(k in config for k in ('validation_manifest','pairing_manifest'))):
        raise ValueError('advanced held_out capture provenance is not supported')
    _stock_baseline(config,run,source)
    if not config.get('native_bridge'):raise ValueError('native_bridge is required for actual candidate decoding/scoring')
    capture=_json(config['capture_manifest']);records=[]
    if method=='local':
        validated=native_codec.validate_activations(config['native_bridge'],config['capture_manifest'],source,threads=config['threads'])
        validation=None
        if config['selection_policy']=='held_out':
            validation=native_codec.validate_activations(config['native_bridge'],config['validation_manifest'],source,threads=config['threads'])
            validate_pairing(_json(config['pairing_manifest']),capture,_json(config['validation_manifest']))
        reader=gguf.GGUFReader(source)
        try:
            tensors={t.name:t for t in reader.tensors}
            for name,t in sorted(run['tensors'].items()):
                if name not in validated['tensors']:raise ValueError('missing local capture site: '+name)
                a=validated['tensors'][name];v=validation['tensors'][name] if validation else None
                for key in [t['stock'],*[c['candidate_id'] for c in t['candidates']]]:
                    c=candidate_for(run,store,name,key)
                    if c.provenance['generator']['family'] not in config['families']:continue
                    score=native_codec.rescore(config['native_bridge'],c,tensors[name],a,validation=v,
                        objective=config['objective'],workspace_bytes=b['workspace_bytes'],threads=config['threads'])
                    records.append({**score,'candidate_id':key,'tensor':name,'method':'local','objective':config['objective'],
                        'capture_sha256':file_digest(config['capture_manifest'])})
        finally:reader.data._mmap.close()
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
        if set(capture)-allowed or not {'schema_version','source_sha256','tokens','layers'}<=set(capture) or capture['schema_version']!=1 or capture['source_sha256']!=run['source_sha256']:
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
    result=dict(schema_version=1,status='PASS',method=method,objective=config['objective'],records=records,
                candidate_manifest_sha256=file_digest(config['candidate_manifest']))
    if method!='local':
        result['reference_identity']=adapter.identity
        result['execution_policy']=policy.identity()
        result['resource_measurements']=monitor.records
        result['loading_estimates']=adapter.loading_estimates
        result['execution_identity_sha256']=adapter.source_id
    result['checksum']=digest(result)
    # Validate serialized size before creating an immutable output directory.
    if len(canonical_json(result))>b['max_output_bytes']:
        raise ValueError('score output byte budget exceeded')
    output=_new_output(config,[source,config['candidate_manifest'],config['capture_manifest']])
    atomic_write(output/'scores.json',canonical_json(result),max_bytes=b['max_output_bytes'])
    return result


def select_config(config,*,resume=None):
    with _run(config) as inputs:return _select(config,*inputs,resume=resume)


def _select(config,run,store,source,*,resume=None):
    from .selector import State,Group,coordinate_descent
    from .hybrid import assemble
    from .kl import NativeKLEvaluator
    from .run_state import selection_identity,open_selection,selection_lock
    b=config['budgets']
    if file_digest(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('selection source does not match candidate provenance')
    selection=_json(config['selection_manifest']);e=_json(config['evaluator'])
    allowed={'schema_version','executable','reference','corpus','native_loader','context','batch','microbatch','max_tokens','timeout','alternatives'}
    if set(e)!=allowed or e['schema_version']!=1:raise ValueError('invalid native evaluator configuration')
    if selection.get('split')!='selection' or selection.get('corpus_sha256')!=file_digest(e['corpus']) or selection.get('reference_sha256')!=file_digest(e['reference']):
        raise ValueError('selection stream/reference provenance mismatch (final-test prohibited)')
    if selection.get('schema_version')!=1 or _json(str(e['reference'])+'.json').get('source_sha256')!=run['source_sha256']:
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
    stock,locked=_stock_baseline(config,run,source)
    evaluator=NativeKLEvaluator(e['executable'],context=e['context'],batch=e['batch'],microbatch=e['microbatch'],max_tokens=e['max_tokens'],timeout=e['timeout'],threads=config['threads'])
    validator=_validator(e['native_loader'],threads=config['threads'])
    for name,key in config['baseline'].items():candidate_for(run,store,name,key)
    groups=[];seen=set()
    if set(e['alternatives'])!=set(config['groups']):raise ValueError('group alternatives mismatch')
    for name,members in sorted(config['groups'].items()):
        if seen.intersection(members):raise ValueError('overlapping selection groups')
        seen.update(members)
        for alternative in e['alternatives'][name]:
            if set(alternative)!=set(members):raise ValueError('group alternative must assign exactly its tensors')
            for tensor,key in alternative.items():candidate_for(run,store,tensor,key)
        groups.append(Group(name,tuple(members),tuple(e['alternatives'][name])))
    if seen!=set(config['baseline']):raise ValueError('groups must cover the complete run tensor allocation')
    identity=selection_identity(config,e,evaluator.policy,stock)
    inputs=[source,config['candidate_manifest'],config['selection_manifest'],config['evaluator'],e['corpus'],e['reference']]
    if stock:inputs.append(stock)
    output=open_selection(config,resume,identity,inputs,_json)
    with selection_lock(output):
        completed=None
        if (output/'selection.json').exists():
            completed=_json(output/'selection.json')
            if completed.get('checksum')!=digest({k:v for k,v in completed.items() if k!='checksum'}) or completed.get('run_identity_sha256')!=identity['checksum']:
                raise ValueError('completed selection identity/checksum mismatch')
            if not (output/'best.gguf').is_file() or file_digest(output/'best.gguf')!=completed['best_gguf_sha256']:
                raise ValueError('completed selection best checksum mismatch')
        initial=State.create(config['baseline'],{'run_identity_sha256':identity['checksum']})
        def evaluate(state):
            with tempfile.TemporaryDirectory(prefix='.evaluation-',dir=output) as temporary:
                path=Path(temporary)/'hybrid.gguf'
                assemble(source,path,store,dict(state.assignment),stock_baseline=stock,
                    native_validate=validator,max_output_bytes=b['max_output_bytes'])
                return evaluator.evaluate(path,e['corpus'],e['reference'],split=config['selection_split'],cache_directory=output/'scores')
        result=coordinate_descent(initial,groups,evaluate,checkpoint=output/'checkpoint.json',
            max_evaluations=b['max_evaluations'],max_passes=config['max_passes'],tolerance=config['tolerance'],split=config['selection_split'])
        if completed is not None:
            if completed['assignment']!=dict(result.state.assignment) or completed['selection_kld']!=result.score:
                raise ValueError('completed result differs from resumed checkpoint')
            return completed
        # A crash after best.gguf publication but before selection.json must also
        # be resumable, without overwriting an unrelated or corrupted artifact.
        with tempfile.TemporaryDirectory(prefix='.finalize-',dir=output) as temporary:
            path=Path(temporary)/'best.gguf'
            best_hash=assemble(source,path,store,dict(result.state.assignment),stock_baseline=stock,
                native_validate=validator,max_output_bytes=b['max_output_bytes'])
            best=output/'best.gguf'
            if best.exists():
                if file_digest(best)!=best_hash:raise ValueError('existing best model collision on resume')
            else:os.link(path,best)
        summary=dict(schema_version=1,status='PASS',run_identity_sha256=identity['checksum'],
            source_sha256=run['source_sha256'],stock_baseline_sha256=locked['baseline_sha256'] if locked else None,
            corpus_partitions=corpus_identity,best_gguf_sha256=best_hash,assignment=dict(result.state.assignment),selection_kld=result.score,
            evaluations=result.evaluations,budget_exhausted=result.budget_exhausted,trace=result.trace,
            warning='Selection objective only; not an untouched-test result or quality conclusion.')
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('--plan',action='store_true')
    parser.add_argument('--resume',metavar='RUN_DIRECTORY',help='resume a strictly matching existing generation or selection run')
    args=parser.parse_args(argv);config=load_config(args.config)
    if args.resume is not None and (config['operation'] not in ('generate','select') or args.plan):
        raise ValueError('--resume requires generate or select without --plan')
    if args.plan:
        result=preflight(config) if config['operation']=='generate' else dict(status='PLAN',operation=config['operation'],resource_bounds=config['resource_bounds'])
    else:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        if config['operation'] in ('select','generate'):
            result=({'select':select_config,'generate':generate}[config['operation']])(config,resume=args.resume)
        else:result={'score':score_config,'assemble':assemble_config,'lock_baseline':lock_baseline_config,'lock_corpora':lock_corpora_config,
                     'validate':validate_config_artifacts}[config['operation']](config)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.TimeoutExpired) as error:
        print('SINQ research: '+str(error),file=sys.stderr);raise SystemExit(1)
