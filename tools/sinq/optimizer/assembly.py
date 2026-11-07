"""Byte-preserving single-file GGUF assembly through the canonical writer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import gguf
from .resources import canonical_json, digest, file_digest, reject_alias, tensor_hashes, observe_phase

PREFIX='beellama.qk_sinq.'
FIELDS={'version','tensors','transform','recipes','producer'}


def _bytes_hash(data):
    with observe_phase('tensor_hash') as measured:
        h=hashlib.sha256()
        raw=data.view(np.uint8).reshape(-1)
        for start in range(0,raw.size,1 << 20):
            block=raw[start:start+(1 << 20)]
            h.update(block)
            if measured is not None:measured['bytes_hashed']+=block.nbytes
        return h.hexdigest()


@observe_phase('canonical_assembly')
def assemble(source, output, store, assignment, *, native_validate, max_output_bytes=1 << 34, stock_baseline=None, threads=1, prepared=None):
    """Canonical output; optional session reuses only guarded INPUT inventories."""
    if prepared is not None:
        prepared.check(source,stock_baseline)
        reject_alias(output,[prepared.baseline_path,prepared.stock_baseline])
        maps=[]
        try:
            return _assemble(source,output,store,assignment,native_validate=native_validate,
                max_output_bytes=max_output_bytes,reader=prepared.reader,maps=maps,
                stock_reader=prepared.stock_reader,manifest=prepared.manifest,
                stock_baseline=prepared.stock_baseline,threads=threads,prepared=prepared)
        finally:
            for mapped in maps:mapped._mmap.close()
    reader=gguf.GGUFReader(source)
    maps=[]; stock_reader=None; manifest=None
    try:
        if stock_baseline is not None:
            from .baseline import load_baseline
            manifest,baseline_path=load_baseline(stock_baseline,source,threads=threads)
            reject_alias(output,[baseline_path,stock_baseline])
            stock_reader=gguf.GGUFReader(baseline_path)
        return _assemble(source,output,store,assignment,native_validate=native_validate,max_output_bytes=max_output_bytes,reader=reader,maps=maps,stock_reader=stock_reader,manifest=manifest,stock_baseline=stock_baseline,threads=threads)
    finally:
        for mapped in maps: mapped._mmap.close()
        if stock_reader is not None: stock_reader.data._mmap.close()
        reader.data._mmap.close()


def _assemble(source, output, store, assignment, *, native_validate, max_output_bytes, reader, maps, stock_reader, manifest, stock_baseline, prepublish_check=None, threads=1, prepared=None):
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
        if present!=FIELDS or reader.fields[PREFIX+'version'].contents()!=2:
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
    with observe_phase('assembly_source_hash'):
        source_hash=prepared.inputs.fingerprint(source) if prepared is not None else file_digest(source)
    with observe_phase('assembly_cache_get'):
        candidates={name:store.get(key,expected={'source_sha256':source_hash,'tensor_name':name}) for name,key in sorted(assignment.items())}
    with observe_phase('assembly_source_hash'):
        source_hashes=({name:manifest['tensors'][name]['source_sha256'] for name in candidates} if prepared is not None
            else tensor_hashes(((name,tensors[name].data) for name in candidates),threads=threads))
    for name,c in candidates.items():
        reject_alias(output,[c.payload]+([c.scales] if c.scales else []))
        t=tensors[name]
        if list(reversed(t.shape.tolist())) != c.provenance['shape']:
            raise ValueError('candidate tensor shape mismatch')
        if t.tensor_type.name!=c.provenance['source_qtype'] or source_hashes[name]!=c.provenance['input_sha256']:
            raise ValueError('candidate original tensor qtype/digest mismatch')
    if manifest is not None:
        stock_tensors={t.name:t for t in stock_reader.tensors}
        for name,c in candidates.items():
            allocation=manifest['tensors'][name]
            if c.provenance['qtype']!=allocation['qtype'] or c.provenance['shape']!=allocation['shape']:
                raise ValueError('candidate differs from baseline allocation')
            if c.provenance['recipe_sha256']!=manifest['recipe_sha256'] or c.provenance['imatrix_sha256']!=manifest['imatrix_sha256']:
                raise ValueError('candidate recipe/imatrix differs from stock baseline')
            if c.provenance['generator']['generator']=='stock' and (c.scales is not None or file_digest(c.payload)!=(manifest['tensors'][name]['stock_sha256'] if prepared is not None else _bytes_hash(stock_tensors[name].data))):
                raise ValueError('stock candidate differs from baseline payload')
    conditioned={n:c for n,c in candidates.items() if c.scales is not None}
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
        with observe_phase('assembly_metadata'):
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
                tensor_names=sorted(conditioned)
                writer.add_uint32(PREFIX+'version',2)
                writer.add_string(PREFIX+'transform','f16_input_diagonal')
                writer.add_array(PREFIX+'tensors',tensor_names)
                writer.add_array(PREFIX+'recipes',[canonical_json(conditioned[n].provenance['generator']).decode('utf-8') for n in tensor_names])
                writer.add_string(PREFIX+'producer','beellama-sinq-optimizer-v2')
        if prepared is None:
            hashes=tensor_hashes(((name,data) for name,_,_,data in items),threads=threads)
        else:
            # CandidateStore.get authenticated content-addressed blobs while its
            # immutable receipts were live; these names are content digests, not
            # metadata guesses. Output readback below still reads EVERY payload.
            hashes={}
            for name,_,_,_ in items:
                base=name.removesuffix('.sinq_scale2')
                candidate=candidates.get(base)
                if candidate is not None:
                    hashes[name]=(candidate.scales if name.endswith('.sinq_scale2') else candidate.payload).name
                else:hashes[name]=manifest['tensors'][name]['stock_sha256']
        for name,shape,q,data in items:
            # Logical shape is already known; uint8 would trigger byte-shape conversion.
            writer.add_tensor_info(name,shape,np.dtype('float16'),data.nbytes,raw_dtype=q)
        with observe_phase('assembly_write') as measured:
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
            for _,_,_,data in items:
                writer.write_tensor_data(data)
            writer.close(); writer=None
            with open(temp,'r+b') as f: os.fsync(f.fileno())
            if measured is not None:measured['bytes_written']+=Path(temp).stat().st_size
        with observe_phase('assembly_readback'):
            check=gguf.GGUFReader(temp)
            try:
                if tensor_hashes(((t.name,t.data) for t in check.tensors),threads=threads) != hashes:
                    raise ValueError('GGUF payload readback mismatch')
                keys={k[len(PREFIX):] for k in check.fields if k.startswith(PREFIX)}
                if keys != (FIELDS if conditioned else set()):
                    raise ValueError('GGUF SINQ namespace mismatch')
            finally:
                check.data._mmap.close()
            del check
        if prepared is not None:prepared.check()
        elif file_digest(source)!=source_hash:
            raise ValueError('source GGUF changed during assembly')
        with observe_phase('assembly_cache_get'):
            for name,c in candidates.items():
                store.get(c.candidate_id,expected={'source_sha256':source_hash,'tensor_name':name})
        with observe_phase('assembly_native_load'):
            native_validate(Path(temp))
        # Validation may execute external code; recheck the entire closure afterwards.
        with observe_phase('assembly_final_integrity'):
            if prepared is not None:prepared.check()
            elif file_digest(source)!=source_hash:
                raise ValueError('source GGUF changed during native validation')
            with observe_phase('assembly_cache_get'):
                for name,c in candidates.items():
                    store.get(c.candidate_id,expected={'source_sha256':source_hash,'tensor_name':name})
            if prepared is not None:prepared.check()
            elif manifest is not None:
                from .baseline import load_baseline
                current,_=load_baseline(stock_baseline,source,threads=threads)
                if current!=manifest: raise ValueError('stock baseline changed during assembly')
            if prepublish_check is not None: prepublish_check()
        os.link(temp,output)
        with observe_phase('assembly_final_integrity'):
            return file_digest(output)
    finally:
        if writer is not None: writer.close()
        if os.path.exists(temp): os.unlink(temp)
