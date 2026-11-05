"""Byte-preserving single-file GGUF assembly through the canonical writer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import gguf
from .resources import digest, file_digest, reject_alias

PREFIX='beellama.qk_sinq.'
FIELDS={'version','tensors','algorithm','strip_width','iterations','producer','numerics'}


def _bytes_hash(data):
    h=hashlib.sha256()
    raw=data.view(np.uint8).reshape(-1)
    for start in range(0,raw.size,1 << 20):
        h.update(raw[start:start+(1 << 20)])
    return h.hexdigest()


def assemble(source, output, store, assignment, *, native_validate, max_output_bytes=1 << 34, stock_baseline=None):
    reader=gguf.GGUFReader(source)
    maps=[]; stock_reader=None; manifest=None
    try:
        if stock_baseline is not None:
            from .baseline import load_baseline
            manifest,baseline_path=load_baseline(stock_baseline,source)
            reject_alias(output,[baseline_path,stock_baseline])
            stock_reader=gguf.GGUFReader(baseline_path)
        return _assemble(source,output,store,assignment,native_validate=native_validate,max_output_bytes=max_output_bytes,reader=reader,maps=maps,stock_reader=stock_reader,manifest=manifest,stock_baseline=stock_baseline)
    finally:
        for mapped in maps: mapped._mmap.close()
        if stock_reader is not None: stock_reader.data._mmap.close()
        reader.data._mmap.close()


def assemble_from_anchor(source, anchor, anchor_manifest, candidate_manifest, output, store, assignment, *, native_validate, max_output_bytes=1 << 34):
    """Assemble from a locked historical conditioned anchor without requantizing it.

    Candidate provenance is checked against the original floating source, while all
    unassigned tensors and metadata originate from the historical mixed-quant anchor.
    The complete conditioned allocation must be explicit so no stale scale pair can
    survive a replacement.
    """
    source,anchor,anchor_manifest,candidate_manifest,output=map(Path,(source,anchor,anchor_manifest,candidate_manifest,output))
    reject_alias(output,[source,anchor,anchor_manifest,candidate_manifest])
    if output.exists(): raise ValueError('hybrid output already exists')
    lock=json.loads(anchor_manifest.read_text(encoding='utf-8'))
    if lock.get('schema_version')!=1 or lock.get('checksum')!=digest({k:v for k,v in lock.items() if k!='checksum'}):
        raise ValueError('historical anchor lock schema/checksum mismatch')
    if Path(lock.get('source_path','')).resolve()!=source.resolve() or Path(lock.get('anchor_path','')).resolve()!=anchor.resolve():
        raise ValueError('historical anchor lock paths differ from assembly inputs')
    source_sha=lock.get('source_sha256');anchor_sha=lock.get('anchor_sha256')
    if file_digest(source)!=source_sha: raise ValueError('historical anchor source SHA-256 mismatch')
    if file_digest(anchor)!=anchor_sha: raise ValueError('historical anchor GGUF SHA-256 mismatch')
    run=json.loads(candidate_manifest.read_text(encoding='utf-8'))
    if run.get('schema_version')!=1 or run.get('checksum')!=digest({k:v for k,v in run.items() if k!='checksum'}):
        raise ValueError('candidate run manifest schema/checksum mismatch')
    if (run.get('source_sha256')!=source_sha or run.get('recipe_sha256')!=lock['recipe_sha256'] or
            run.get('imatrix_sha256')!=lock['imatrix_sha256']):
        raise ValueError('candidate run differs from historical anchor source/recipe/imatrix')
    if (candidate_manifest.parent/run.get('cache','')).resolve()!=store.root:
        raise ValueError('candidate store does not belong to the locked run manifest')
    if not isinstance(lock.get('tensors'),dict) or not lock['tensors']:
        raise ValueError('historical anchor lock has no tensor inventory')
    if not isinstance(assignment,dict): raise ValueError('historical anchor assignment must be a mapping')
    maps=[];source_reader=gguf.GGUFReader(source);anchor_reader=gguf.GGUFReader(anchor)
    try:
        source_tensors={t.name:t for t in source_reader.tensors}
        anchor_tensors={t.name:t for t in anchor_reader.tensors}
        records=lock['tensors']
        conditioned={name for name,record in records.items() if record.get('companion_bytes',0)>0}
        if set(source_tensors)!=set(records): raise ValueError('historical anchor lock/source tensor inventory mismatch')
        expected_anchor_names=set(source_tensors)|{name+'.sinq_scale2' for name in conditioned}
        if set(anchor_tensors)!=expected_anchor_names: raise ValueError('historical anchor tensor/sidecar inventory mismatch')
        if set(assignment)!=conditioned:
            raise ValueError('all conditioned anchor tensors must be assigned exactly once')
        if set(run.get('tensors',{}))!=conditioned:
            raise ValueError('candidate run tensor set differs from conditioned anchor allocation')
        prefix='beellama.qk_sinq.'
        present={key[len(prefix):] for key in anchor_reader.fields if key.startswith(prefix)}
        if present!=FIELDS or anchor_reader.fields[prefix+'version'].contents()!=1:
            raise ValueError('historical anchor SINQ namespace is incomplete')
        if set(anchor_reader.fields[prefix+'tensors'].contents())!=conditioned:
            raise ValueError('historical anchor SINQ tensor manifest mismatch')
        semantics={'algorithm':lock['algorithm'],'strip_width':lock['width'],
                   'iterations':lock['iterations'],'numerics':lock['numerics']}
        for key,value in semantics.items():
            if anchor_reader.fields[prefix+key].contents()!=value:
                raise ValueError('historical anchor metadata mismatch: '+key)
        if not callable(native_validate): raise ValueError('native loader validation callback is required')
        lock_sha=file_digest(anchor_manifest);candidate_manifest_sha=file_digest(candidate_manifest)
        for name,record in records.items():
            original=source_tensors[name];base=anchor_tensors[name]
            shape=record.get('shape')
            if list(reversed(original.shape.tolist()))!=shape or original.tensor_type.name!=record.get('source_qtype'):
                raise ValueError('historical source tensor qtype/shape mismatch: '+name)
            if list(reversed(base.shape.tolist()))!=shape or base.tensor_type.name!=record.get('qtype'):
                raise ValueError('historical anchor tensor qtype/shape mismatch: '+name)
            companion=anchor_tensors.get(name+'.sinq_scale2')
            if name in conditioned:
                if companion is None or companion.tensor_type!=gguf.GGMLQuantizationType.F16 or list(reversed(companion.shape.tolist()))!=[shape[1]] or companion.data.nbytes!=record['companion_bytes']:
                    raise ValueError('historical anchor scale companion mismatch: '+name)
                key=assignment[name];run_record=run['tensors'][name]
                permitted={run_record['stock'],*[item['candidate_id'] for item in run_record['candidates']]}
                if run_record.get('anchor_candidate_id')!=record.get('candidate_id') or key not in permitted:
                    raise ValueError('candidate is not the locked anchor or a permitted run alternative: '+name)
                candidate=store.get(key,expected={'source_sha256':source_sha,'tensor_name':name,
                    'recipe_sha256':lock['recipe_sha256'],'imatrix_sha256':lock['imatrix_sha256']})
                provenance=candidate.provenance
                if provenance['generator']['family'] not in ('ordinary','awq_weight_activation','awq_activation_only') or candidate.scales is None:
                    raise ValueError('historical anchor assignment is not a conditioned candidate: '+name)
                if provenance['shape']!=shape or provenance['qtype']!=record['qtype'] or provenance['source_qtype']!=record['source_qtype']:
                    raise ValueError('candidate differs from historical anchor allocation: '+name)
                candidate_semantics=provenance['sinq']
                if any(candidate_semantics.get(field)!=value for field,value in (
                    ('algorithm',semantics['algorithm']),('strip_width',semantics['strip_width']),
                    ('iterations',semantics['iterations']),('numerics',semantics['numerics']))):
                    raise ValueError('candidate SINQ semantics differ from historical anchor: '+name)
                if key==record.get('candidate_id') and (file_digest(candidate.payload)!=record['packed_sha256'] or
                        file_digest(candidate.scales)!=record['companion_sha256']):
                    raise ValueError('original anchor candidate differs from locked payload/companion: '+name)
            elif companion is not None or record.get('companion_sha256') is not None:
                raise ValueError('unexpected unconditioned anchor companion: '+name)
        def prepublish_check():
            if (file_digest(anchor)!=anchor_sha or file_digest(anchor_manifest)!=lock_sha or
                    file_digest(candidate_manifest)!=candidate_manifest_sha):
                raise ValueError('historical anchor or input lock changed during assembly')
        return _assemble(source,output,store,assignment,native_validate=native_validate,max_output_bytes=max_output_bytes,
            reader=source_reader,maps=maps,stock_reader=anchor_reader,manifest=None,stock_baseline=None,
            prepublish_check=prepublish_check)
    finally:
        for mapped in maps: mapped._mmap.close()
        anchor_reader.data._mmap.close();source_reader.data._mmap.close()


def _assemble(source, output, store, assignment, *, native_validate, max_output_bytes, reader, maps, stock_reader, manifest, stock_baseline, prepublish_check=None):
    """Validate callback must raise on native loader failure; publication follows it.

    Assignment replaces complete payload/sidecar pairs. Existing conditioned tensors
    must be explicitly assigned, preventing accidental reuse of stale companions.
    """
    source,output=Path(source),Path(output)
    reject_alias(output,[source])
    if output.exists():
        raise ValueError('hybrid output already exists')
    if not callable(native_validate):
        raise ValueError('native loader validation callback is required')
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError('only little-endian single-file GGUF is supported')
    if any(k.startswith('split.') for k in reader.fields):
        raise ValueError('split GGUF is unsupported')
    unknown={k[len(PREFIX):] for k in reader.fields if k.startswith(PREFIX)}-FIELDS
    if unknown:
        raise ValueError('unknown SINQ namespace keys')
    tensors={t.name:t for t in reader.tensors}
    if len(tensors)!=len(reader.tensors) or not set(assignment)<=set(tensors):
        raise ValueError('duplicate or unmatched tensor name')
    existing={n[:-len('.sinq_scale2')] for n in tensors if n.endswith('.sinq_scale2')}
    present={k[len(PREFIX):] for k in reader.fields if k.startswith(PREFIX)}
    if present or existing:
        if present!=FIELDS or reader.fields[PREFIX+'version'].contents()!=1:
            raise ValueError('incomplete source SINQ namespace')
        declared=reader.fields[PREFIX+'tensors'].contents()
        if not isinstance(declared,list) or len(declared)!=len(set(declared)) or set(declared)!=existing:
            raise ValueError('stale source SINQ tensor manifest')
        for name in existing:
            companion=tensors[name+'.sinq_scale2']
            if name not in tensors or companion.tensor_type!=gguf.GGMLQuantizationType.F16 or companion.shape.tolist()!=[int(tensors[name].shape[0])]:
                raise ValueError('orphan or invalid source scale companion')
    if not existing<=set(assignment):
        raise ValueError('existing conditioned tensors require explicit candidate assignment')
    source_hash=file_digest(source)
    candidates={name:store.get(key,expected={'source_sha256':source_hash,'tensor_name':name}) for name,key in sorted(assignment.items())}
    for name,c in candidates.items():
        reject_alias(output,[c.payload]+([c.scales] if c.scales else []))
        t=tensors[name]
        if list(reversed(t.shape.tolist())) != c.provenance['shape']:
            raise ValueError('candidate tensor shape mismatch')
        if t.tensor_type.name!=c.provenance['source_qtype'] or _bytes_hash(t.data)!=c.provenance['input_sha256']:
            raise ValueError('candidate original tensor qtype/digest mismatch')
    if manifest is not None:
        stock_tensors={t.name:t for t in stock_reader.tensors}
        for name,c in candidates.items():
            allocation=manifest['tensors'][name]
            if c.provenance['qtype']!=allocation['qtype'] or c.provenance['shape']!=allocation['shape']:
                raise ValueError('candidate differs from baseline allocation')
            if c.provenance['recipe_sha256']!=manifest['recipe_sha256'] or c.provenance['imatrix_sha256']!=manifest['imatrix_sha256']:
                raise ValueError('candidate recipe/imatrix differs from stock baseline')
            if c.provenance['generator']['family']=='stock' and (c.scales is not None or file_digest(c.payload)!=_bytes_hash(stock_tensors[name].data)):
                raise ValueError('stock candidate differs from baseline payload')
    conditioned={n:c for n,c in candidates.items() if c.scales is not None}
    semantics={(c.provenance['sinq']['strip_width'],c.provenance['sinq']['iterations'],c.provenance['sinq']['algorithm'],c.provenance['sinq']['numerics']) for c in conditioned.values()}
    if len(semantics)>1:
        raise ValueError('mixed global SINQ width/iterations/semantics')
    compatibility={(c.provenance['recipe_sha256'],c.provenance['imatrix_sha256']) for c in candidates.values()}
    if len(compatibility)>1:
        raise ValueError('mixed recipe/imatrix provenance')
    items=[]
    for t in (stock_reader if stock_reader is not None else reader).tensors:
        if t.name.endswith('.sinq_scale2'):
            continue
        c=candidates.get(t.name)
        if c:
            data=np.memmap(c.payload,dtype=np.uint8,mode='r')
            maps.append(data)
            q=gguf.GGMLQuantizationType[c.provenance['qtype']]
            shape=c.provenance['shape']
        else:
            data=t.data; q=t.tensor_type; shape=list(reversed(t.shape.tolist()))
        items.append((t.name,shape,q,data))
        if c and c.scales:
            scales=np.memmap(c.scales,dtype='<f2',mode='r')
            maps.append(scales)
            items.append((t.name+'.sinq_scale2',[shape[1]],gguf.GGMLQuantizationType.F16,scales))
    if sum(x[3].nbytes+32 for x in items)+source.stat().st_size-reader.data_offset > max_output_bytes:
        raise ValueError('hybrid output budget exceeded')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.hybrid-',suffix='.gguf',dir=output.parent); os.close(fd)
    writer=None
    try:
        metadata_reader=stock_reader if stock_reader is not None else reader
        arch=metadata_reader.fields['general.architecture'].contents()
        writer=gguf.GGUFWriter(temp,arch)
        for key,field in metadata_reader.fields.items():
            if key.startswith('GGUF.') or key.startswith(PREFIX) or key in ('general.architecture','general.file_type','general.quantization_version'):
                continue
            writer.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
        # Let the native loader infer mixed allocation from actual tensor types,
        # rather than preserving an original floating-only file classification.
        if any(gguf.GGML_QUANT_SIZES[q][0]>1 for _,_,q,_ in items):
            writer.add_uint32('general.quantization_version',gguf.GGML_QUANT_VERSION)
        if conditioned:
            width,iterations,algorithm,numerics=next(iter(semantics))
            writer.add_uint32(PREFIX+'version',1)
            tensor_names=sorted(conditioned)
            if manifest is None and stock_reader is not None and PREFIX+'tensors' in metadata_reader.fields:
                anchor_names=metadata_reader.fields[PREFIX+'tensors'].contents()
                if (not isinstance(anchor_names,list) or len(anchor_names)!=len(set(anchor_names)) or
                        set(anchor_names)!=set(conditioned)):
                    raise ValueError('historical anchor SINQ tensor declaration differs from conditioned payloads')
                tensor_names=anchor_names
            writer.add_array(PREFIX+'tensors',tensor_names)
            writer.add_string(PREFIX+'algorithm',algorithm)
            writer.add_uint32(PREFIX+'strip_width',width)
            writer.add_uint32(PREFIX+'iterations',iterations)
            writer.add_string(PREFIX+'producer','beellama-sinq-research-v1')
            writer.add_string(PREFIX+'numerics',numerics)
        hashes={}
        for name,shape,q,data in items:
            # Logical shape is already known; uint8 would trigger byte-shape conversion.
            writer.add_tensor_info(name,shape,np.dtype('float16'),data.nbytes,raw_dtype=q)
            hashes[name]=_bytes_hash(data)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
        for _,_,_,data in items:
            writer.write_tensor_data(data)
        writer.close(); writer=None
        with open(temp,'r+b') as f: os.fsync(f.fileno())
        check=gguf.GGUFReader(temp)
        try:
            if {t.name:_bytes_hash(t.data) for t in check.tensors} != hashes:
                raise ValueError('GGUF payload readback mismatch')
            keys={k[len(PREFIX):] for k in check.fields if k.startswith(PREFIX)}
            if keys != (FIELDS if conditioned else set()):
                raise ValueError('GGUF SINQ namespace mismatch')
        finally:
            check.data._mmap.close()
        del check
        if file_digest(source)!=source_hash:
            raise ValueError('source GGUF changed during assembly')
        for name,c in candidates.items():
            store.get(c.candidate_id,expected={'source_sha256':source_hash,'tensor_name':name})
        native_validate(Path(temp))
        # Validation may execute external code; recheck the entire closure afterwards.
        if file_digest(source)!=source_hash:
            raise ValueError('source GGUF changed during native validation')
        for name,c in candidates.items():
            store.get(c.candidate_id,expected={'source_sha256':source_hash,'tensor_name':name})
        if manifest is not None:
            from .baseline import load_baseline
            current,_=load_baseline(stock_baseline,source)
            if current!=manifest: raise ValueError('stock baseline changed during assembly')
        if prepublish_check is not None: prepublish_check()
        os.link(temp,output)
        return file_digest(output)
    finally:
        if writer is not None: writer.close()
        if os.path.exists(temp): os.unlink(temp)
