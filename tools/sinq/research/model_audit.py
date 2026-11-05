"""Audit an assembled model against its locked historical anchor and candidates."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import gguf
from .resources import digest, file_digest

PREFIX='beellama.qk_sinq.'
SINQ_FIELDS={'version','tensors','algorithm','strip_width','iterations','producer','numerics'}
_METADATA_EXCEPTIONS={'GGUF.version','GGUF.tensor_count','GGUF.kv_count','general.file_type',PREFIX+'producer'}


def _array_sha(data):
    h=hashlib.sha256();raw=data.view(np.uint8).reshape(-1)
    for start in range(0,raw.size,1<<20):h.update(raw[start:start+(1<<20)])
    return h.hexdigest()


def _simple(value):
    if isinstance(value,np.ndarray):return [_simple(x) for x in value.tolist()]
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,bytes):return {'bytes_hex':value.hex()}
    if isinstance(value,(list,tuple)):return [_simple(x) for x in value]
    if isinstance(value,dict):return {str(k):_simple(v) for k,v in sorted(value.items(),key=lambda item:str(item[0]))}
    return value


def _inventory(reader):
    return {tensor.name:{'qtype':tensor.tensor_type.name,
            'shape':list(reversed(tensor.shape.tolist())),
            'bytes':int(tensor.data.nbytes),'sha256':_array_sha(tensor.data)}
            for tensor in reader.tensors}


def audit_assembled_model(model,anchor,anchor_manifest,candidate_manifest,store,assignment,*,recipe):
    """Verify complete weight/sidecar identity, anchor inheritance and GGUF metadata."""
    model,anchor,anchor_manifest,candidate_manifest=map(Path,(model,anchor,anchor_manifest,candidate_manifest))
    lock=json.loads(anchor_manifest.read_text(encoding='utf-8'))
    if lock.get('schema_version')!=1 or lock.get('checksum')!=digest({k:v for k,v in lock.items() if k!='checksum'}):
        raise ValueError('anchor audit lock schema/checksum mismatch')
    if file_digest(anchor)!=lock.get('anchor_sha256') or Path(lock.get('anchor_path','')).resolve()!=anchor.resolve():
        raise ValueError('anchor audit GGUF identity mismatch')
    run=json.loads(candidate_manifest.read_text(encoding='utf-8'))
    if run.get('schema_version')!=1 or run.get('checksum')!=digest({k:v for k,v in run.items() if k!='checksum'}):
        raise ValueError('anchor audit candidate manifest schema/checksum mismatch')
    if (run.get('source_sha256')!=lock.get('source_sha256') or run.get('recipe_sha256')!=lock.get('recipe_sha256') or
            run.get('imatrix_sha256')!=lock.get('imatrix_sha256') or
            (candidate_manifest.parent/run.get('cache','')).resolve()!=store.root):
        raise ValueError('anchor audit candidate source/recipe/imatrix/cache mismatch')
    records=lock.get('tensors')
    if not isinstance(records,dict) or set(run.get('tensors',{}))!={n for n,r in records.items() if r.get('companion_bytes',0)>0}:
        raise ValueError('anchor audit allocation differs from candidate manifest')
    conditioned=set(run['tensors'])
    if not isinstance(assignment,dict) or set(assignment)!=conditioned:
        raise ValueError('anchor audit requires one assignment for every conditioned tensor')
    source_sha=lock['source_sha256'];recipe_sha=lock['recipe_sha256'];imatrix_sha=lock['imatrix_sha256']
    anchor_reader=gguf.GGUFReader(anchor);model_reader=gguf.GGUFReader(model)
    try:
        anchor_inventory=_inventory(anchor_reader);model_inventory=_inventory(model_reader)
        base_names=set(records)
        if set(n for n in anchor_inventory if not n.endswith('.sinq_scale2'))!=base_names:
            raise ValueError('anchor audit base tensor inventory differs from lock')
        sidecar_names={n+'.sinq_scale2' for n in conditioned}
        if set(anchor_inventory)!=base_names|sidecar_names or set(model_inventory)!=base_names|sidecar_names:
            raise ValueError('anchor audit tensor/sidecar closure mismatch')
        for name,record in records.items():
            old=anchor_inventory[name];companion=name+'.sinq_scale2'
            if (old['qtype']!=record['qtype'] or old['shape']!=record['shape'] or
                    old['bytes']!=record['packed_bytes'] or old['sha256']!=record['packed_sha256']):
                raise ValueError('locked anchor payload mismatch: '+name)
            if name in conditioned:
                old_scale=anchor_inventory[companion]
                if (old_scale['qtype']!='F16' or old_scale['shape']!=[record['shape'][1]] or
                        old_scale['bytes']!=record['companion_bytes'] or old_scale['sha256']!=record['companion_sha256']):
                    raise ValueError('locked anchor companion mismatch: '+name)
                row=run['tensors'][name]
                key=assignment[name]
                permitted={row['stock'],*(item['candidate_id'] for item in row['candidates'])}
                if row.get('anchor_candidate_id')!=record.get('candidate_id') or key not in permitted:
                    raise ValueError('assignment is not in immutable candidate pool: '+name)
                candidate=store.get(key,expected={'source_sha256':source_sha,'tensor_name':name,
                    'recipe_sha256':recipe_sha,'imatrix_sha256':imatrix_sha})
                provenance=candidate.provenance
                if (provenance['generator']['family'] not in ('ordinary','awq_weight_activation','awq_activation_only') or
                        candidate.scales is None or provenance['shape']!=record['shape'] or
                        provenance['source_qtype']!=record['source_qtype'] or provenance['qtype']!=record['qtype']):
                    raise ValueError('selected candidate differs from anchor codec allocation: '+name)
                semantics=provenance['sinq']
                for field,expected in (('algorithm',lock['algorithm']),('strip_width',lock['width']),
                                       ('iterations',lock['iterations']),('numerics',lock['numerics'])):
                    if semantics.get(field)!=expected:raise ValueError('candidate anchor semantics mismatch: '+name)
                if key==record.get('candidate_id') and (
                    file_digest(candidate.payload)!=record['packed_sha256'] or
                    file_digest(candidate.scales)!=record['companion_sha256']):
                    raise ValueError('O assignment differs from locked anchor payload: '+name)
                expected_payload=file_digest(candidate.payload)
                expected_scale=file_digest(candidate.scales)
                if model_inventory[name]['sha256']!=expected_payload or model_inventory[companion]['sha256']!=expected_scale:
                    raise ValueError('assembled candidate payload/companion mismatch: '+name)
                if (model_inventory[name]['qtype']!=record['qtype'] or model_inventory[name]['shape']!=record['shape'] or
                        model_inventory[companion]['qtype']!='F16' or model_inventory[companion]['shape']!=[record['shape'][1]]):
                    raise ValueError('assembled candidate qtype/shape mismatch: '+name)
            else:
                if companion in anchor_inventory or record.get('companion_sha256') is not None:
                    raise ValueError('unexpected unconditioned anchor companion: '+name)
                if model_inventory[name]!=old:
                    raise ValueError('unassigned historical anchor tensor was changed: '+name)
        anchor_fields=anchor_reader.fields;model_fields=model_reader.fields
        anchor_keys={k[len(PREFIX):] for k in anchor_fields if k.startswith(PREFIX)}
        model_keys={k[len(PREFIX):] for k in model_fields if k.startswith(PREFIX)}
        if anchor_keys!=SINQ_FIELDS or model_keys!=SINQ_FIELDS:
            raise ValueError('assembled SINQ namespace is incomplete')
        for suffix,value in (('version',1),('algorithm',lock['algorithm']),('strip_width',lock['width']),
                             ('iterations',lock['iterations']),('numerics',lock['numerics'])):
            if _simple(model_fields[PREFIX+suffix].contents())!=value:
                raise ValueError('assembled SINQ metadata mismatch: '+suffix)
        if set(model_fields[PREFIX+'tensors'].contents())!=conditioned:
            raise ValueError('assembled SINQ tensor declaration differs from sidecar closure')
        metadata_differences=[]
        for key in sorted(set(anchor_fields)|set(model_fields)):
            if key in _METADATA_EXCEPTIONS:continue
            before=_simple(anchor_fields[key].contents()) if key in anchor_fields else None
            after=_simple(model_fields[key].contents()) if key in model_fields else None
            if before!=after:metadata_differences.append(key)
        if metadata_differences:raise ValueError('unexpected anchor metadata changes: '+','.join(metadata_differences))
        metadata_exception_differences=[]
        for key in sorted(_METADATA_EXCEPTIONS):
            before=_simple(anchor_fields[key].contents()) if key in anchor_fields else None
            after=_simple(model_fields[key].contents()) if key in model_fields else None
            if before!=after:metadata_exception_differences.append(key)
        qtype_counts={}
        for name in base_names:qtype_counts[model_inventory[name]['qtype']]=qtype_counts.get(model_inventory[name]['qtype'],0)+1
        params=sum(math.prod(record['shape']) for record in records.values())
        payload_bytes=sum(item['bytes'] for item in model_inventory.values())
        sidecar_bytes=sum(model_inventory[n+'.sinq_scale2']['bytes'] for n in conditioned)
        main_bytes=payload_bytes-sidecar_bytes
        model_bytes=model.stat().st_size
        changed=[name for name in sorted(base_names) if model_inventory[name]!=anchor_inventory[name]]
        changed_companions=[name+'.sinq_scale2' for name in sorted(conditioned)
                            if model_inventory[name+'.sinq_scale2']!=anchor_inventory[name+'.sinq_scale2']]
        inventory_sha=digest(model_inventory);anchor_inventory_sha=digest(anchor_inventory)
        return {'schema_version':1,'kind':'audited-anchor-assembly','status':'PASS','recipe':recipe,
          'model_path':str(model.resolve()),'model_sha256':file_digest(model),'model_bytes':model_bytes,
          'anchor_path':str(anchor.resolve()),'anchor_sha256':lock['anchor_sha256'],
          'anchor_manifest_sha256':file_digest(anchor_manifest),'candidate_manifest_sha256':file_digest(candidate_manifest),
          'source_sha256':source_sha,'recipe_sha256':recipe_sha,'imatrix_sha256':imatrix_sha,
          'tensor_inventory_sha256':inventory_sha,'anchor_tensor_inventory_sha256':anchor_inventory_sha,
          'identical_to_anchor_by_tensor_inventory':inventory_sha==anchor_inventory_sha,
          'base_tensor_count':len(base_names),'conditioned_tensor_count':len(conditioned),
          'total_gguf_tensor_count':len(model_inventory),'parameter_count':params,
          'qtype_counts':dict(sorted(qtype_counts.items())),'changed_tensors':changed,
          'changed_companions':changed_companions,'assignment_candidate_ids':dict(sorted(assignment.items())),
          'main_tensor_payload_bytes':main_bytes,'sidecar_bytes':sidecar_bytes,'total_tensor_payload_bytes':payload_bytes,
          'gguf_header_metadata_bytes':model_bytes-payload_bytes,'sidecar_bpw':sidecar_bytes*8/params,
          'payload_bpw':payload_bytes*8/params,'file_bpw':model_bytes*8/params,
          'writer_metadata_exception_differences':metadata_exception_differences,
          'metadata_equal_except_writer_fields':True}
    finally:
        anchor_reader.data._mmap.close();model_reader.data._mmap.close()
