"""Fail-closed selection run identities and crash-released exclusive ownership."""
from contextlib import contextmanager
import os
from pathlib import Path
from .resources import atomic_write,canonical_json,digest,file_digest,reject_alias
from .implementation import implementation_fingerprint,native_runtime_fingerprint


def selection_identity(config,evaluator,policy,stock_baseline=None):
    paths={'source':config['source_gguf'],'candidates':config['candidate_manifest'],
        'selection':config['selection_manifest'],'evaluator_config':config['evaluator'],
        'corpus':evaluator['corpus'],'reference':evaluator['reference'],
        'reference_manifest':str(evaluator['reference'])+'.json',
        'evaluator_binary':evaluator['executable'],'native_loader':evaluator['native_loader']}
    if stock_baseline is not None:paths['stock_baseline_manifest']=stock_baseline
    if 'corpus_partitions_manifest' in config:paths['corpus_partitions']=config['corpus_partitions_manifest']
    for i,stage in enumerate(config.get('stages',[])):
        if 'scores' in stage:paths['stage_scores_'+str(i)]=stage['scores']
    implementation=implementation_fingerprint()
    body={'schema_version':2,'kind':'sinq-selection-run','config_sha256':digest(config),
        'bindings':{name:file_digest(path) for name,path in sorted(paths.items())},'policy':policy,
        'implementation_sha256':implementation['checksum'],'implementation':implementation,
        'native_runtime':native_runtime_fingerprint({'evaluator':evaluator['executable'],'loader':evaluator['native_loader']})}
    body['checksum']=digest(body)
    return body


def open_run(config,resume,identity,inputs,read_json):
    output=Path(config['output_directory']).resolve()
    reject_alias(output,inputs)
    if resume is not None:
        if Path(resume).resolve()!=output or not output.is_dir():
            raise ValueError('resume directory must be the existing configured output directory')
        path=output/'run-identity.json'
        if not path.is_file() or read_json(path)!=identity:
            raise ValueError('research resume identity mismatch; configuration, code, dependencies or inputs changed')
    else:
        if output.exists():raise FileExistsError('immutable research output already exists; use --resume RUN_DIRECTORY')
        output.mkdir(parents=True,exist_ok=False)
        atomic_write(output/'run-identity.json',canonical_json(identity),max_bytes=1<<20)
    return output


open_selection=open_run


@contextmanager
def selection_lock(output,*,name='selection'):
    """OS locks release on process death, unlike a stale sentinel after a crash."""
    if name not in ('selection','generation'):raise ValueError('unknown run lock kind')
    path=Path(output)/(name+'.lock')
    if path.is_symlink() or (path.exists() and (path.stat().st_size>1 or path.stat().st_nlink>1)):
        raise ValueError('unsafe selection lock artifact')
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        locked=False
        try:
            if os.name=='nt':
                import msvcrt
                try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as error:raise RuntimeError('selection run is already active') from error
            else:
                import fcntl
                try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except OSError as error:raise RuntimeError('selection run is already active') from error
            locked=True
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
