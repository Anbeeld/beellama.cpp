"""Complete stock allocations, independently verified with the ordinary native codec."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import gguf
from .resources import atomic_write, canonical_json, digest, file_digest, file_digests, tensor_hashes, reject_alias

PREFIX='beellama.qk_sinq.'


def _metadata(reader):
    # Native quantize updates classification/provenance, not graph/tokenizer semantics.
    return {k: {'types':[int(t) for t in f.types], 'value':f.contents()}
            for k,f in reader.fields.items() if not k.startswith(('GGUF.','quantize.'))
            and k not in ('general.file_type','general.quantization_version')}


def _inspect(source, baseline, recipe, *, threads=1):
    readers=[]
    try:
        for path in (source,baseline):
            r=gguf.GGUFReader(path); readers.append(r)
            if r.endianess!=gguf.GGUFEndian.LITTLE or any(k.startswith(('split.',PREFIX)) for k in r.fields) or any(t.name.endswith('.sinq_scale2') for t in r.tensors):
                raise ValueError('baseline requires untransformed single-file GGUF')
            if r.fields['general.architecture'].contents()!='qwen35':
                raise ValueError('baseline requires Qwen35')
        original,stock=readers
        names={t.name for t in original.tensors}
        if len(names)!=len(original.tensors) or names!=set(recipe):
            raise ValueError('complete recipe must name every original tensor exactly once')
        if any(not isinstance(q,str) or q not in gguf.GGMLQuantizationType.__members__ for q in recipe.values()):
            raise ValueError('recipe requires exact uppercase GGUF qtypes')
        if any(t.tensor_type.name not in ('F32','F16','BF16') for t in original.tensors):
            raise ValueError('original source must contain only floating tensors')
        if _metadata(original)!=_metadata(stock):
            raise ValueError('source/baseline graph or tokenizer metadata mismatch')
        tensors={t.name:t for t in stock.tensors}
        if len(tensors)!=len(stock.tensors) or set(tensors)!=names:
            raise ValueError('source/baseline tensor names mismatch')
        hashes=tensor_hashes(((str(index)+':'+t.name,t.data) for index,r in enumerate(readers) for t in r.tensors),threads=threads)
        records={}
        for t in original.tensors:
            b=tensors[t.name]
            if t.shape.tolist()!=b.shape.tolist() or b.tensor_type.name!=recipe[t.name]:
                raise ValueError('baseline shape or recipe allocation mismatch: '+t.name)
            records[t.name]={'shape':list(reversed(t.shape.tolist())), 'source_qtype':t.tensor_type.name,
                'qtype':b.tensor_type.name,'source_sha256':hashes['0:'+t.name],'stock_sha256':hashes['1:'+b.name]}
        return records,digest(_metadata(original))
    finally:
        for r in readers: r.data._mmap.close()


def _run(command, output, *, timeout, max_output_bytes, threads):
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),CUDA_VISIBLE_DEVICES='')
    # No pipes or unbounded log accumulation, even on a failing native command.
    with subprocess.Popen(command,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as process:
        start=time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic()-start>timeout:
                    raise TimeoutError('stock baseline native reencode timed out')
                if output.exists() and output.stat().st_size>max_output_bytes:
                    raise ValueError('stock baseline output budget exceeded')
                time.sleep(.02)
            if process.returncode:
                raise ValueError('ordinary native reencode failed (exit %d)'%process.returncode)
            if not output.is_file() or output.stat().st_size>max_output_bytes:
                raise ValueError('stock baseline output missing or budget exceeded')
        except BaseException:
            if process.poll() is None: process.kill()
            process.wait()
            raise


def lock_baseline(source, baseline, recipe, output, *, quantizer, native_validate,
                  imatrix=None, workspace_bytes=64<<20, max_output_bytes=1<<34,
                  threads=None, timeout=3600):
    """Reencode once, compare every payload, then publish an immutable v1 JSON lock."""
    source,baseline,output,quantizer=map(lambda p:Path(p).resolve(),(source,baseline,output,quantizer))
    imatrix=Path(imatrix).resolve() if imatrix is not None else None
    reject_alias(output,[source,baseline,quantizer]+([imatrix] if imatrix else []))
    if output.exists(): raise ValueError('baseline manifest already exists')
    if not callable(native_validate): raise ValueError('native loader validation is required')
    threads=(os.cpu_count() or 1) if threads is None else threads
    if (type(threads)!=int or not 1<=threads<=1024 or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0
            or type(workspace_bytes)!=int or workspace_bytes<1<<20 or type(max_output_bytes)!=int or max_output_bytes<=0):
        raise ValueError('invalid baseline resource bounds')
    if baseline.stat().st_size>max_output_bytes: raise ValueError('baseline output budget exceeded')
    recipe=dict(recipe)
    paths={'source':source,'baseline':baseline,'quantizer':quantizer}
    if imatrix: paths['imatrix']=imatrix
    hashes=file_digests(paths,threads=threads)
    records,metadata_sha256=_inspect(source,baseline,recipe,threads=threads)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stock-reencode-',dir=output.parent) as directory:
        directory=Path(directory); types=directory/'types.txt'; generated=directory/'verified.gguf'
        if any(any(c.isspace() for c in name) or '=' in name for name in recipe):
            raise ValueError('tensor name cannot be represented in native type file')
        types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(recipe.items())),encoding='utf-8')
        command=[str(quantizer),'--pure','--tensor-type-file',str(types),'--max-buffer-size',str(workspace_bytes//(1<<20))]
        if imatrix: command+=['--imatrix',str(imatrix)]
        # A quantized default activates native per-tensor patterns; the complete
        # allocation is checked below, so no unspecified fallback is accepted.
        command += [str(source),str(generated),'Q8_0',str(threads)]
        _run(command,generated,timeout=timeout,max_output_bytes=max_output_bytes,threads=threads)
        regenerated,metadata=_inspect(source,generated,recipe,threads=threads)
        if regenerated!=records or metadata!=metadata_sha256:
            raise ValueError('stock payload differs from independent ordinary native reencode')
        native_validate(baseline)
        if file_digests(paths,threads=threads)!=hashes:
            raise ValueError('baseline inputs changed during verification')
        manifest={'version':2,'kind':'sinq-stock-baseline','verification':'native-reencode-exact',
            'source_sha256':hashes['source'],'baseline_sha256':hashes['baseline'],
            'baseline_path':os.path.relpath(baseline,output.parent),'recipe':recipe,'recipe_sha256':digest(recipe),
            'imatrix_sha256':hashes.get('imatrix'),'quantizer_sha256':hashes['quantizer'],
            'metadata_sha256':metadata_sha256,'tensors':records}
        manifest['checksum']=digest(manifest)
        atomic_write(output,canonical_json(manifest),max_bytes=64<<20)
        return manifest


def load_baseline(manifest_path, source, *, threads=1):
    """Validate the entire closure, using bounded independent hash workers."""
    if isinstance(threads,bool) or not isinstance(threads,int) or not 1<=threads<=1024:
        raise ValueError('baseline threads must be an integer in 1..1024')
    manifest_path=Path(manifest_path).resolve(); source=Path(source).resolve()
    if manifest_path.stat().st_size>64<<20: raise ValueError('baseline manifest budget exceeded')
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('duplicate baseline JSON field')
            result[key]=value
        return result
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'),object_pairs_hook=unique_object)
    keys={'version','kind','verification','source_sha256','baseline_sha256','baseline_path','recipe','recipe_sha256','imatrix_sha256','quantizer_sha256','metadata_sha256','tensors','checksum'}
    if not isinstance(manifest,dict) or set(manifest)!=keys:
        raise ValueError('invalid baseline manifest fields')
    body={k:v for k,v in manifest.items() if k!='checksum'}
    if digest(body)!=manifest['checksum'] or manifest['version']!=2 or manifest['kind']!='sinq-stock-baseline' or manifest['verification']!='native-reencode-exact':
        raise ValueError('invalid baseline manifest checksum/version/verification')
    for key in ('checksum','source_sha256','baseline_sha256','recipe_sha256','quantizer_sha256','metadata_sha256','imatrix_sha256'):
        value=manifest[key]
        if key=='imatrix_sha256' and value is None: continue
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
            raise ValueError('invalid baseline digest: '+key)
    if not isinstance(manifest['baseline_path'],str) or Path(manifest['baseline_path']).is_absolute():
        raise ValueError('baseline path must be relative to manifest')
    baseline=(manifest_path.parent/manifest['baseline_path']).resolve()
    paths={'source_sha256':source,'baseline_sha256':baseline}
    expected={key:manifest[key] for key in paths}
    if file_digests(paths,threads=threads)!=expected:
        raise ValueError('baseline/source checkpoint digest mismatch')
    if digest(manifest['recipe'])!=manifest['recipe_sha256']:
        raise ValueError('baseline recipe digest mismatch')
    records,metadata=_inspect(source,baseline,manifest['recipe'],threads=threads)
    if records!=manifest['tensors'] or metadata!=manifest['metadata_sha256']:
        raise ValueError('baseline tensor or metadata closure mismatch')
    if file_digests(paths,threads=threads)!=expected:
        raise ValueError('baseline/source changed during readback')
    return manifest,baseline
