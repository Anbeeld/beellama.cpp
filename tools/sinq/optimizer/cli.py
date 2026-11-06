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
    from .resources import file_digest,peak_rss_bytes,digest
    inputs={key:file_digest(config[key]) for key in ('source_gguf','native_bridge','native_loader','candidate_manifest',
        'capture_manifest','activation_manifest','validation_manifest','pairing_manifest','stock_baseline_manifest',
        'corpus_partitions_manifest','selection_manifest','evaluator','artifact_manifest') if key in config and Path(config[key]).is_file()}
    imatrix=config.get('native_quantizer',{}).get('imatrix_path')
    if imatrix:inputs['imatrix']=file_digest(imatrix)
    output=Path(config['output_directory']) if 'output_directory' in config else None
    artifacts={}
    if stage!='plan' and output is not None and output.is_dir():
        for path in output.rglob('*'):
            if path.is_symlink():raise ValueError('unexpected symlink in stage output')
            if path.is_file():
                if len(artifacts)>=config['budgets']['max_files']:raise ValueError('stage output file budget exceeded')
                artifacts[path.relative_to(output).as_posix()]=dict(bytes=path.stat().st_size,sha256=file_digest(path))
    elif stage=='assemble':
        path=Path(config['output_gguf']);artifacts[path.name]=dict(bytes=path.stat().st_size,sha256=file_digest(path))
    tensors=result.get('tensors',{})
    return dict(stage=stage,config_sha256=digest(config),input_hashes=inputs,artifacts=artifacts,
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
    from .resources import observe_caches
    with observe_caches() as caches:
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
    print(json.dumps({**result,'stage_summary':_stage_summary(args.stage,config,result,started,caches)},sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.TimeoutExpired) as error:
        print('SINQ optimizer: '+str(error),file=sys.stderr);raise SystemExit(1)
