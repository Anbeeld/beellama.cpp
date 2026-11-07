"""Independent complete allocation and v2 companion audit, without historical anchors."""
import json
import math
from pathlib import Path
import gguf
import numpy as np
from .resources import digest,file_digest,tensor_hashes
from .schemas import parameters

PREFIX='beellama.qk_sinq.'
SINQ_FIELDS={'version','transform','tensors','recipes','producer'}


def audit_assembled_model(model,source,stock_manifest,store,assignment,*,threads=1,prepared=None):
    from .baseline import load_baseline
    if prepared is not None:
        prepared.check(source,stock_manifest)
        manifest=prepared.manifest
        original,baseline=prepared.reader,prepared.stock_reader
    else:
        manifest,stock=load_baseline(stock_manifest,source,threads=threads)
        original,baseline=gguf.GGUFReader(source),gguf.GGUFReader(stock)
    # The output always receives an independent complete inventory/hash audit.
    output=gguf.GGUFReader(model)
    try:
        maps=[{t.name:t for t in r.tensors} for r in (original,baseline,output)]
        src,base,result=maps
        if any(len(m)!=len(r.tensors) for m,r in zip(maps,(original,baseline,output))):raise ValueError('duplicate tensor inventory')
        if not set(assignment)<=src.keys() or set(base)!=set(src):raise ValueError('source/stock/assignment allocation mismatch')
        candidates={n:store.get(k,expected=dict(source_sha256=manifest['source_sha256'],tensor_name=n,
            recipe_sha256=manifest['recipe_sha256'],imatrix_sha256=manifest['imatrix_sha256'])) for n,k in assignment.items()}
        conditioned={n:c for n,c in candidates.items() if c.scales is not None}
        if set(result)!=set(src)|{n+'.sinq_scale2' for n in conditioned}:raise ValueError('output companion inventory mismatch')
        hashes=tensor_hashes(((n,t.data) for n,t in result.items()),threads=threads)
        for name,source_tensor in src.items():
            t=result[name];record=manifest['tensors'][name]
            if t.shape.tolist()!=source_tensor.shape.tolist() or t.tensor_type.name!=record['qtype']:raise ValueError('output recipe qtype/shape mismatch: '+name)
            candidate=candidates.get(name)
            expected=file_digest(candidate.payload) if candidate else record['stock_sha256']
            if hashes[name]!=expected:raise ValueError('selected/unselected output payload mismatch: '+name)
            if name in conditioned:
                scale=result[name+'.sinq_scale2']
                if scale.tensor_type.name!='F16' or scale.shape.tolist()!=[int(t.shape[0])] or hashes[scale.name]!=file_digest(candidate.scales):raise ValueError('output scale pair mismatch')
                for i in range(0,len(scale.data),65536):
                    chunk=scale.data[i:i+65536]
                    if not np.all(np.isfinite(chunk)&(chunk>0)):raise ValueError('invalid F16 companion')
        keys={k[len(PREFIX):] for k in output.fields if k.startswith(PREFIX)}
        if keys!=(SINQ_FIELDS if conditioned else set()):raise ValueError('output SINQ namespace mismatch')
        if conditioned:
            if output.fields[PREFIX+'version'].contents()!=2 or output.fields[PREFIX+'transform'].contents()!='f16_input_diagonal':raise ValueError('unsupported SINQ version/transform')
            names=output.fields[PREFIX+'tensors'].contents();recipes=output.fields[PREFIX+'recipes'].contents()
            if names!=sorted(conditioned) or len(recipes)!=len(names):raise ValueError('output recipe coverage/order mismatch')
            for name,text in zip(names,recipes):
                if parameters(json.loads(text))!=conditioned[name].provenance['generator']:raise ValueError('output parameter descriptor mismatch')
        inventory={n:dict(shape=list(reversed(t.shape.tolist())),qtype=t.tensor_type.name,bytes=int(t.n_bytes),sha256=hashes[n]) for n,t in result.items()}
        count=sum(math.prod(t.shape.tolist()) for t in original.tensors);payload=sum(t.n_bytes for t in output.tensors)
        return dict(schema_version=2,status='PASS',source_sha256=manifest['source_sha256'],recipe_sha256=manifest['recipe_sha256'],
            imatrix_sha256=manifest['imatrix_sha256'],output_sha256=file_digest(model),tensor_inventory=inventory,
            tensor_inventory_sha256=digest(inventory),base_tensor_count=len(src),conditioned_tensor_count=len(conditioned),
            payload_bpw=payload*8/count,file_bpw=Path(model).stat().st_size*8/count,parameter_count=count)
    finally:
        output.data._mmap.close()
        if prepared is not None:prepared.check()
        else:
            for r in (original,baseline):r.data._mmap.close()
