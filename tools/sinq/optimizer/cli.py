"""Strict offline SINQ optimizer. Heavy dependencies are imported only by their stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'gguf-py'))
from .schemas import load_config


def _stage_summary(stage,config,result,started,caches):
    import hashlib
    from .resources import peak_rss_bytes,digest,observe_phase
    limits=config.get('budgets',{})
    remaining_bytes=limits.get('max_output_bytes',8<<20)
    remaining_files=limits.get('max_files',20)

    def small_file(path,expected=None,identity=None):
        nonlocal remaining_bytes,remaining_files
        path=Path(path)
        if path.is_symlink():raise ValueError('unexpected symlink in summary inventory')
        size=path.stat().st_size
        if remaining_files<1 or size>remaining_bytes:raise ValueError('summary inventory resource budget exceeded')
        # Limit the read as well as the stat check: a growing file cannot cause
        # unbounded summary I/O. No process-global stat/digest cache is involved.
        count=0;body=[];hasher=hashlib.sha256()
        with observe_phase('file_hash') as measured:
            with path.open('rb') as stream:
                while block:=stream.read(min(1<<20,remaining_bytes-count+1)):
                    count+=len(block)
                    if count>remaining_bytes:raise ValueError('summary inventory byte budget exceeded')
                    hasher.update(block)
                    if measured is not None:measured['bytes_hashed']+=len(block)
                    if identity is not None:body.append(block)
            sha=hasher.hexdigest()
        if expected is not None and sha!=expected:raise ValueError('summary manifest digest mismatch')
        if identity is not None:
            value=json.loads(b''.join(body))
            if value.get('checksum')!=identity or digest({k:v for k,v in value.items() if k!='checksum'})!=identity:
                raise ValueError('summary run identity mismatch')
        remaining_files-=1;remaining_bytes-=count
        return dict(path=str(path),bytes=count,sha256=sha,fresh_content_hash=True,hash_scope='fresh-bounded-file-hash')

    inputs={};input_evidence={}
    if 'source_gguf' in config:
        sha=result.get('source_sha256')
        if sha is not None:
            inputs['source_gguf']=sha
            input_evidence['source_gguf']=dict(path=str(config['source_gguf']),sha256=sha,fresh_content_hash=False,
                hash_scope='live-stage-verification-event-not-fresh-hash')
        elif stage=='plan' and Path(config['source_gguf']).is_file():
            # Compatibility for preflight callers without a returned identity.
            evidence=small_file(config['source_gguf'])
            inputs['source_gguf']=evidence['sha256'];input_evidence['source_gguf']=evidence
    for key in ('candidate_manifest','capture_manifest','activation_manifest','validation_manifest','pairing_manifest',
            'stock_baseline_manifest','corpus_partitions_manifest','selection_manifest','evaluator','artifact_manifest'):
        if key in config and Path(config[key]).is_file():
            evidence=small_file(config[key],expected=result.get(key+'_sha256'))
            inputs[key]=evidence['sha256'];input_evidence[key]=evidence
    output=Path(config['output_directory']) if 'output_directory' in config else None
    artifacts={}
    # Only stage-owned publications, never caches, logs, native runtimes,
    # references, or arbitrary files left in a resumed run directory.
    names={'generate':('run-manifest.json','run-identity.json'),
        'select':('selection.json','checkpoint.json','run-identity.json'),
        'score':('scores.json',)}.get(stage,())
    primary={'generate':'run-manifest.json','select':'selection.json','score':'scores.json'}.get(stage)
    if output is not None:
        for name in names:
            path=output/name
            if path.is_symlink():raise ValueError('unexpected symlink in summary inventory')
            if path.is_file():
                artifacts[name]=small_file(path,expected=digest(result) if name==primary else None,
                    identity=result.get('run_identity_sha256') if name=='run-identity.json' else None)
    model=None;published=None
    if stage=='select' and output is not None:
        model=output/'best.gguf';published=result.get('best_gguf_sha256')
    elif stage=='assemble':
        model=Path(config['output_gguf']);published=result.get('output_sha256')
    if model is not None and published is not None:
        if model.is_symlink():raise ValueError('unexpected symlink in summary inventory')
        artifacts[model.name]=dict(path=str(model),bytes=model.stat().st_size,sha256=published,
            fresh_content_hash=False,hash_scope='published-output-evidence-not-fresh-hash')
    tensors=result.get('tensors',{})
    return dict(stage=stage,config_sha256=digest(config),input_hashes=inputs,input_hash_evidence=input_evidence,artifacts=artifacts,
        inventory_scope='Explicit stage publications and bounded input metadata only; not a directory census or fresh model verification',
        selected_tensors=len(result.get('assignment',tensors)),
        candidate_count=sum(len(r['candidates']) for r in tensors.values()),skipped_candidate_count=sum(len(r.get('skipped',[])) for r in tensors.values()),
        parameter_space=config.get('resolved_candidates'),execution_policy={k:config[k] for k in ('threads','device','reference_dtype','cuda_devices','memory_policy') if k in config},
        process_peak_rss_bytes=peak_rss_bytes(),memory_scope='Python process lifetime high-water RSS; native workspace and CUDA phase measurements are recorded separately',
        elapsed_seconds=time.monotonic()-started,cache=caches,evaluations=result.get('evaluations'),budget_exhausted=result.get('budget_exhausted',False),
        scorer_assumption=result.get('warning','Surrogate scores are frozen-context estimates; not untouched-test quality evidence.' if stage=='score' else None))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='stage',required=True)
    for name in ('plan','generate','score','select','assemble','validate'):
        sub=commands.add_parser(name)
        sub.add_argument('--config',required=True)
        if name in ('generate','select'):sub.add_argument('--resume',metavar='RUN_DIR')
    args=parser.parse_args(argv);config=load_config(args.config,args.stage);started=time.monotonic()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from .resources import observe_caches,observe_performance
    with observe_caches() as caches,observe_performance() as performance:
        if args.stage=='plan':
            from .generate import preflight
            result=preflight(config)
        elif args.stage=='generate':
            from .generate import generate
            result=generate(config,resume=args.resume)
        else:
            from .stages import score_config,select_config,assemble_config,validate_config_artifacts
            dispatch={'score':score_config,'select':select_config,'assemble':assemble_config,'validate':validate_config_artifacts}
            result=dispatch[args.stage](config,**({'resume':args.resume} if args.stage=='select' else {}))
        summary=_stage_summary(args.stage,config,result,started,caches)
    summary['performance']=performance
    summary['performance_scope']='Inclusive wall seconds (nested phases overlap); bytes_hashed count direct file/tensor hash input, including repeated checks. Includes stage-summary hashing.'
    print(json.dumps({**result,'stage_summary':summary},sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.TimeoutExpired) as error:
        print('SINQ optimizer: '+str(error),file=sys.stderr);raise SystemExit(1)
