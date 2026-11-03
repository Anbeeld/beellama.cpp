"""Byte-preserving single-file GGUF assembly through the canonical writer."""
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np
import gguf
from .resources import file_digest, reject_alias

PREFIX='beellama.qk_sinq.'
FIELDS={'version','tensors','algorithm','strip_width','iterations','producer','numerics'}


def _bytes_hash(data):
    h=hashlib.sha256()
    raw=data.view(np.uint8).reshape(-1)
    for start in range(0,raw.size,1 << 20):
        h.update(raw[start:start+(1 << 20)])
    return h.hexdigest()


def assemble(source, output, store, assignment, *, native_validate, max_output_bytes=1 << 34):
    reader=gguf.GGUFReader(source)
    maps=[]
    try:
        return _assemble(source,output,store,assignment,native_validate=native_validate,max_output_bytes=max_output_bytes,reader=reader,maps=maps)
    finally:
        for mapped in maps: mapped._mmap.close()
        reader.data._mmap.close()


def _assemble(source, output, store, assignment, *, native_validate, max_output_bytes, reader, maps):
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
    conditioned={n:c for n,c in candidates.items() if c.scales is not None}
    semantics={(c.provenance['sinq']['strip_width'],c.provenance['sinq']['iterations'],c.provenance['sinq']['algorithm'],c.provenance['sinq']['numerics']) for c in conditioned.values()}
    if len(semantics)>1:
        raise ValueError('mixed global SINQ width/iterations/semantics')
    compatibility={(c.provenance['recipe_sha256'],c.provenance['imatrix_sha256']) for c in candidates.values()}
    if len(compatibility)>1:
        raise ValueError('mixed recipe/imatrix provenance')
    items=[]
    for t in reader.tensors:
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
        arch=reader.fields['general.architecture'].contents()
        writer=gguf.GGUFWriter(temp,arch)
        for key,field in reader.fields.items():
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
            writer.add_array(PREFIX+'tensors',sorted(conditioned))
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
        os.link(temp,output)
        return file_digest(output)
    finally:
        if writer is not None: writer.close()
        if os.path.exists(temp): os.unlink(temp)
