"""Immutable native packed tensor/F16 diagonal bundles (row-major [output,input])."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import gguf
from .resources import atomic_write, canonical_json, digest, file_digest

_REQUIRED = {'source_sha256','tensor_name','shape','source_qtype','qtype','input_sha256','recipe_sha256','imatrix_sha256','codec','sinq','generator','activation','producer'}


def _valid_scales(values):
    return all(np.all(np.isfinite(values[i:i+65536])) and np.all(values[i:i+65536]>0) for i in range(0,len(values),65536))


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA256')
    return value


def validate_provenance(p):
    if set(p) != _REQUIRED:
        raise ValueError('candidate provenance fields mismatch')
    canonical_json(p)
    for k in ('source_sha256','input_sha256','recipe_sha256'):
        sha(p[k])
    if p['imatrix_sha256'] is not None:
        sha(p['imatrix_sha256'])
    for k in ('binary_sha256','source_sha256'):
        sha(p['producer'][k])
    if not p['codec'] or p['codec'].get('implementation') != 'ggml':
        raise ValueError('actual GGML codec provenance required')
    if not isinstance(p['tensor_name'], str) or not p['tensor_name'] or p['tensor_name'].endswith('.sinq_scale2'):
        raise ValueError('invalid base tensor name')
    shape = p['shape']
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('expected positive row-major matrix shape')
    try:
        q = gguf.GGMLQuantizationType[p['qtype']]
        gguf.GGMLQuantizationType[p['source_qtype']]
        block, size = gguf.GGML_QUANT_SIZES[q]
    except (KeyError, TypeError) as e:
        raise ValueError('unsupported qtype') from e
    if shape[1] % block:
        raise ValueError('qtype block shape mismatch')
    s = p['sinq']
    if s is not None:
        if set(s) != {'strip_width','iterations','strength','algorithm','numerics'}:
            raise ValueError('invalid SINQ provenance')
        if s['strip_width'] not in (32,64,128,256,512) or shape[1] % s['strip_width'] or s['iterations'] not in (4,8,16):
            raise ValueError('invalid SINQ width/iterations')
        if not np.isfinite(s['strength']) or not 0 <= s['strength'] <= 1 or s['algorithm'] not in ('sinkhorn_log','identity') or not s['numerics']:
            raise ValueError('invalid SINQ semantics')
    g = p['generator']
    if not isinstance(g, dict) or not g.get('family'):
        raise ValueError('generator provenance required')
    if (g['family']=='stock') != (s is None):
        raise ValueError('stock/conditioned semantics mismatch')
    if g['family'] not in ('stock','ordinary','awq_weight_activation','awq_activation_only'):
        raise ValueError('unsupported generator family')
    for field in ('alpha','lambda'):
        if field in g and (not isinstance(g[field],(int,float)) or not np.isfinite(g[field]) or not 0 <= g[field] <= 1):
            raise ValueError('invalid generator parameter')
    if g['family'] not in ('stock','ordinary') and p['activation'] is None:
        raise ValueError('activation-bound generator requires provenance')
    if p['activation'] is not None:
        for k in ('source_sha256','split_sha256'):
            sha(p['activation'][k])
    return shape[0]*shape[1]//block*size


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    provenance: dict
    payload: Path
    scales: Path | None


class CandidateStore:
    def __init__(self, root, *, max_bytes=1 << 30, max_files=10000):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_files = max_files

    def _budget(self, extra):
        count=0; total=0
        for path in self.root.rglob('*') if self.root.exists() else ():
            if path.is_file():
                count+=1; total+=path.stat().st_size
                if count+4 > self.max_files or total+extra > self.max_bytes:
                    raise ValueError('cache resource budget exceeded')
        if extra>self.max_bytes or self.max_files<4:
            raise ValueError('cache resource budget exceeded')

    def put_files(self, provenance, payload_path, scale_path=None):
        """Bounded mapped input: no full-tensor Python byte copy."""
        paths=[Path(payload_path)]+([Path(scale_path)] if scale_path is not None else [])
        if any(p.stat().st_size>self.max_bytes or not p.stat().st_size for p in paths):
            raise ValueError('input artifact byte budget exceeded')
        maps=[np.memmap(p,dtype=np.uint8,mode='r') for p in paths]
        try:
            return self.put(provenance,maps[0],maps[1] if len(maps)>1 else None)
        finally:
            for mapped in maps: mapped._mmap.close()

    def put(self, provenance, payload, scales=None):
        self.root.mkdir(parents=True,exist_ok=True)
        lock=self.root/'.write-lock'
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as error:
            raise RuntimeError('cache writer active or interrupted; explicit lock recovery required') from error
        try:
            return self._put_locked(provenance,payload,scales)
        finally:
            os.close(fd)
            lock.unlink()

    def _put_locked(self, provenance, payload, scales=None):
        p = json.loads(canonical_json(provenance))
        expected = validate_provenance(p)
        if len(payload) != expected:
            raise ValueError('native payload byte/shape/qtype mismatch')
        if (p['sinq'] is None) != (scales is None):
            raise ValueError('candidate must contain matching payload/scale pair')
        if scales is not None:
            values = np.frombuffer(scales, dtype='<f2')
            if len(scales) != p['shape'][1]*2 or not _valid_scales(values):
                raise ValueError('invalid serialized F16 diagonal')
        desc = {'schema_version':1,'provenance':p,'payload':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload),'scales':None if scales is None else hashlib.sha256(scales).hexdigest(),'scale_bytes':0 if scales is None else len(scales)}
        key = digest(desc)
        identity=self.root/'identities'/f'{digest(p)}.json'
        if identity.exists() and identity.read_bytes()!=canonical_json({'candidate_id':key}):
            raise ValueError('identical candidate provenance has different native bytes')
        path = self.root/'tensors'/f'{key}.json'
        if path.exists():
            self.get(key)
            return key
        self._budget(len(payload)+(len(scales) if scales is not None else 0)+len(canonical_json(desc)))
        for h, data in ((desc['payload'],payload),(desc['scales'],scales)):
            if data is not None:
                atomic_write(self.root/'blobs'/h,data,max_bytes=self.max_bytes)
        atomic_write(identity,canonical_json({'candidate_id':key}),max_bytes=self.max_bytes)
        atomic_write(path,canonical_json(desc),max_bytes=self.max_bytes)
        self.get(key)
        return key

    def get(self, key, *, expected=None):
        sha(key)
        path = self.root/'tensors'/f'{key}.json'
        if path.stat().st_size > min(self.max_bytes, 1 << 20):
            raise ValueError('descriptor budget exceeded')
        desc = json.loads(path.read_bytes())
        if desc.get('schema_version') != 1 or digest(desc) != key:
            raise ValueError('candidate descriptor hash/schema mismatch')
        p = desc['provenance']
        size = validate_provenance(p)
        identity=self.root/'identities'/f'{digest(p)}.json'
        if not identity.exists() or identity.stat().st_size>256 or identity.read_bytes()!=canonical_json({'candidate_id':key}):
            raise ValueError('candidate semantic identity closure mismatch')
        if desc['payload_bytes']!=size or desc['scale_bytes']!=(p['shape'][1]*2 if p['sinq'] else 0) or (desc['scales'] is None)!=(p['sinq'] is None):
            raise ValueError('candidate descriptor pair/shape mismatch')
        if size>self.max_bytes:
            raise ValueError('candidate payload budget exceeded')
        if expected is not None and any(p.get(k) != v for k,v in expected.items()):
            raise ValueError('candidate provenance mismatch')
        paths=[]
        for label, count in (('payload',size),('scales',p['shape'][1]*2 if p['sinq'] else 0)):
            h = desc[label]
            if h is None:
                if count: raise ValueError('missing scale companion')
                paths.append(None)
                continue
            sha(h)
            blob=self.root/'blobs'/h
            if blob.stat().st_size != count or file_digest(blob) != h:
                raise ValueError('candidate blob checksum/size mismatch')
            paths.append(blob)
        if paths[1] is not None:
            scales=np.memmap(paths[1],dtype='<f2',mode='r')
            if not _valid_scales(scales):
                raise ValueError('invalid F16 companion')
        return Candidate(key,p,*paths)
