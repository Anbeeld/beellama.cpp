"""Immutable native packed tensor/F16 diagonal bundles (row-major [output,input])."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import copy
import tempfile
import struct
from contextlib import contextmanager
import numpy as np
import gguf
from .resources import canonical_json, digest, file_digest

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


class _WriterLock:
    """Persistent lock inode; kernel ownership, not file existence, is the lock."""
    def __init__(self, root):
        if (root/'.write-lock').exists():
            raise RuntimeError('legacy cache writer lock requires explicit recovery')
        self.file = None
        self.handle = None
        try:
            if os.name == 'nt':
                self.api = _win_api()
                self.handle = self.api.CreateFileW(str((root/'.writer-lock').resolve()), 0xC0000000, 1, None, 4, 0x80, None)
                if self.handle == self.api.invalid:
                    self.handle = None
                    raise OSError('cache writer ownership unavailable')
            else:
                import fcntl
                self.file = open(root/'.writer-lock', 'a+b')
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.close()
            raise RuntimeError('cache writer active or ownership unavailable') from error

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None
        if self.file is not None:
            self.file.close()
            self.file = None


def _win_api():
    import ctypes as c
    from ctypes import wintypes as w
    api = c.WinDLL('kernel32', use_last_error=True)
    class Overlapped(c.Structure):
        _fields_ = [('Internal', c.c_size_t), ('InternalHigh', c.c_size_t),
                    ('Offset', w.DWORD), ('OffsetHigh', w.DWORD), ('hEvent', w.HANDLE)]
    api.CreateFileW.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE]
    api.CreateFileW.restype = w.HANDLE
    api.CloseHandle.argtypes = [w.HANDLE]
    api.CreateEventW.argtypes = [w.LPVOID,w.BOOL,w.BOOL,w.LPCWSTR]
    api.CreateEventW.restype = w.HANDLE
    api.ResetEvent.argtypes = [w.HANDLE]
    api.ReadDirectoryChangesW.argtypes = [w.HANDLE,w.LPVOID,w.DWORD,w.BOOL,w.DWORD,c.POINTER(w.DWORD),c.POINTER(Overlapped),w.LPVOID]
    api.GetOverlappedResult.argtypes = [w.HANDLE,c.POINTER(Overlapped),c.POINTER(w.DWORD),w.BOOL]
    api.CancelIoEx.argtypes = [w.HANDLE,c.POINTER(Overlapped)]
    api.invalid = c.c_void_p(-1).value
    api.Overlapped = Overlapped
    return api


class _WindowsIndex:
    """Mandatory immutable file guards plus loss-detecting subtree name events.

    No stat-only hash receipts: every represented file remains deny-write/delete
    guarded. Directory guards prevent rebinding the watched subtree.
    """
    def __init__(self, root):
        import ctypes as c
        self.root = root.resolve()
        self.api = _win_api()
        self.guards = {}
        self.file_count = 0
        self.directory_count = 0
        self.watch = None
        self.event = None
        self.pending = False
        self.owned_events = {}
        self.buffer = c.create_string_buffer(65536)
        self.overlapped = self.api.Overlapped()
        try:
            self.watch = self._open(self.root, 1, 7, 0x42000000)
            self.event = self.api.CreateEventW(None, True, False, None)
            if not self.event:
                raise OSError(c.get_last_error(), 'cannot create cache directory event')
            self.overlapped.hEvent = self.event
            self.guard(self.root)
            self._arm()
        except BaseException:
            self.close()
            raise

    def _open(self, path, access, sharing, flags):
        import ctypes as c
        handle = self.api.CreateFileW(str(path), access, sharing, None, 3, flags, None)
        if handle == self.api.invalid:
            raise OSError(c.get_last_error(), 'cannot guard cache artifact', str(path))
        return handle

    def guard(self, path):
        resolved = path.resolve()
        if resolved != path.absolute() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise OSError('cache aliases cannot be immutably indexed')
        path = resolved
        if path not in self.guards:
            directory = path.is_dir()
            self.guards[path] = self._open(path, 0x80 if directory else 0x80000000,
                                          3 if directory else 1, 0x02000000 if directory else 0x80)
            self.directory_count += int(directory)
            self.file_count += int(not directory)

    def allow(self, path, actions):
        name = str(path.resolve().relative_to(self.root)).replace('/', '\\').casefold()
        self.owned_events.setdefault(name, set()).update(actions)

    def _arm(self):
        import ctypes as c
        self.api.ResetEvent(self.event)
        if not self.api.ReadDirectoryChangesW(self.watch, self.buffer, len(self.buffer), True, 3, None, c.byref(self.overlapped), None):
            raise OSError(c.get_last_error(), 'cannot watch cache namespace')
        self.pending = True

    def changed(self):
        import ctypes as c
        from ctypes import wintypes as w
        count = w.DWORD()
        if not self.api.GetOverlappedResult(self.watch, c.byref(self.overlapped), c.byref(count), False):
            if c.get_last_error() == 996:
                return False
            raise OSError(c.get_last_error(), 'cache namespace watch failed')
        self.pending = False
        changed = count.value == 0  # Overflow is never accepted as no change.
        offset = 0
        while offset < count.value:
            next_offset, action, length = struct.unpack_from('<III', self.buffer, offset)
            name = self.buffer.raw[offset+12:offset+12+length].decode('utf-16-le').casefold()
            allowed = self.owned_events.get(name, set())
            if action not in allowed:
                changed = True
            else:
                allowed.discard(action)
                if not allowed:
                    self.owned_events.pop(name, None)
            if not next_offset:
                break
            offset += next_offset
        self._arm()
        return changed

    def close(self):
        import ctypes as c
        if self.watch is not None:
            if self.pending:
                self.api.CancelIoEx(self.watch, c.byref(self.overlapped))
                # Complete cancellation before freeing OVERLAPPED/buffer.
                from ctypes import wintypes as w
                count = w.DWORD()
                self.api.GetOverlappedResult(self.watch, c.byref(self.overlapped), c.byref(count), True)
                self.pending = False
            self.api.CloseHandle(self.watch)
            self.watch = None
        for handle in self.guards.values():
            self.api.CloseHandle(handle)
        self.guards.clear()
        if self.event is not None:
            self.api.CloseHandle(self.event)
            self.event = None


class CandidateStore:
    """Native CAS. Session/counter state is per instance, not thread-safe.

    index_stats counters are cumulative; files/bytes/guarded_files are current
    gauges. hash_calls/hash_bytes count streaming file checks, not in-memory
    provenance/input SHA computations. Receipt state never outlives a session.
    """
    def __init__(self, root, *, max_bytes=1 << 30, max_files=10000):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._depth = 0
        self._dirty = False
        self._session_identity = None
        self._owner = None
        self._index = None
        self._receipts = {}
        self._hash_receipts = {}
        self._files = {}
        self.index_stats = dict(census_scans=0, census_files=0, census_bytes=0,
                                files=0, bytes=0, hash_calls=0, hash_bytes=0,
                                receipt_hits=0, receipt_misses=0, hash_reuses=0, fallbacks=0,
                                guarded_files=0, namespace_rebuilds=0,
                                fallback_reason=None)

    def _fallback(self, reason):
        if self._index is not None:
            self._index.close()
        self._index = None
        self._receipts.clear()
        self._hash_receipts.clear()
        self.index_stats['guarded_files'] = 0
        self.index_stats['fallbacks'] += 1
        self.index_stats['fallback_reason'] = str(reason)

    @contextmanager
    def verified_session(self):
        """Exclusive, reentrant run-local verification; no persistent receipts.

        Windows mandatory guards enable reuse. Other platforms (and guard/watch
        failures) retain full validation and a fresh census for every put/get.
        The persistent .writer-lock is OS-owned and survives safely after kill;
        a legacy .write-lock is deliberately never recovered automatically.
        """
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._owner = _WriterLock(self.root)
        self._depth = 1
        try:
            info = self.root.stat()
            self._session_identity = (info.st_dev, info.st_ino)
            if os.name == 'nt':
                try:
                    self._index = _WindowsIndex(self.root)
                except OSError as error:
                    self._fallback(error)
            else:
                self._fallback('platform has no mandatory immutable Win32 guards')
            self._census()
            self._check_budget(0, 0)
            yield self
        finally:
            if self._index is not None:
                self._index.close()
            self._index = None
            self._receipts.clear()
            self._hash_receipts.clear()
            self.index_stats['guarded_files'] = 0
            self._depth = 0
            self._dirty = False
            self._session_identity = None
            self._owner.close()
            self._owner = None

    def _census(self):
        files = {}
        total = 0
        self.index_stats['census_scans'] += 1
        try:
            if self._index is not None:
                self._index.guard(self.root)
            for path in self.root.rglob('*'):
                if path.is_symlink():
                    if self._index is not None:
                        raise OSError('cache aliases cannot be immutably indexed')
                if self._index is not None and path != self.root/'.writer-lock':
                    self._index.guard(path)
                if path.is_file():
                    size = path.stat().st_size
                    files[path.absolute()] = size
                    total += size
                    if len(files) > self.max_files or total > self.max_bytes:
                        self._files = files
                        self.index_stats['census_files'] += len(files)
                        self.index_stats['census_bytes'] += total
                        self._update_totals()
                        raise ValueError('cache resource budget exceeded')
        except OSError as error:
            if self._index is None:
                raise
            self._fallback(error)
            return self._census()
        self._files = files
        self.index_stats['census_files'] += len(files)
        self.index_stats['census_bytes'] += sum(files.values())
        self._update_totals()

    def _update_totals(self):
        self.index_stats['files'] = len(self._files)
        self.index_stats['bytes'] = sum(self._files.values())
        self.index_stats['guarded_files'] = (self._index.file_count if self._index is not None else 0)

    def _sync(self):
        if self._depth:
            try:
                info = self.root.stat()
            except OSError as error:
                raise RuntimeError('cache directory identity lost; reopen session') from error
            if (info.st_dev, info.st_ino) != self._session_identity:
                raise RuntimeError('cache directory identity changed; reopen session')
        if self._dirty:
            self._receipts.clear()
            self._hash_receipts.clear()
            self._census()
            self._check_budget(0, 0)
            self._dirty = False
        if self._index is not None:
            try:
                if self._index.changed():
                    self.index_stats['namespace_rebuilds'] += 1
                    self._receipts.clear()
                    self._hash_receipts.clear()
                    self._census()
                    self._check_budget(0, 0)
                    # Events during scanning are checked at the next boundary;
                    # new files never receive receipts before being guarded.
            except OSError as error:
                self._fallback(error)
                self._census()

    def _check_budget(self, extra, files):
        if (self.index_stats['files'] + files > self.max_files or
                self.index_stats['bytes'] + extra > self.max_bytes):
            raise ValueError('cache resource budget exceeded')

    def _budget(self, extra, files=4):
        if self._index is None:
            self._census()
        else:
            self._sync()
        self._check_budget(extra, files)

    def _hash(self, path):
        path = path.resolve()
        if self._index is not None and path in self._hash_receipts:
            self.index_stats['hash_reuses'] += 1
            return self._hash_receipts[path]
        self.index_stats['hash_calls'] += 1
        self.index_stats['hash_bytes'] += path.stat().st_size
        result = file_digest(path)
        if self._index is not None and path in self._index.guards:
            self._hash_receipts[path] = result
        return result

    def _record_file(self, path):
        path = path.absolute()
        size = path.stat().st_size
        old = self._files.get(path)
        self._files[path] = size
        self.index_stats['files'] += int(old is None)
        self.index_stats['bytes'] += size - (old or 0)
        self.index_stats['guarded_files'] = self._index.file_count if self._index is not None else 0

    def _publish(self, path, data):
        if len(data) > self.max_bytes:
            raise ValueError('artifact byte budget exceeded')
        # Track exact atomic-publish names, not timestamp changes or broad
        # '.pending-*' exemptions. Unexpected names invalidate the index.
        for parent in reversed(path.parents):
            if parent == self.root or self.root not in parent.parents:
                continue
            if not parent.exists():
                if self._index is not None:
                    self._index.allow(parent, {1})
                parent.mkdir(exist_ok=True)
                self._guard(parent)
        if path.exists():
            self._guard(path)
            if path.stat().st_size != len(data) or self._hash(path) != hashlib.sha256(data).hexdigest():
                raise ValueError('immutable artifact collision')
            self._record_file(path)
            return
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
        temporary = Path(temporary)
        if self._index is not None:
            self._index.allow(temporary, {1, 2})
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if self._index is not None:
                self._index.allow(path, {1})
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass  # Always validate the actual published file under guard.
        finally:
            temporary.unlink()
        self._guard(path)
        if path.stat().st_size != len(data) or self._hash(path) != hashlib.sha256(data).hexdigest():
            raise ValueError('artifact readback checksum mismatch')
        self._record_file(path)

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
        if self._depth:
            self._sync()
            try:
                return self._put_locked(provenance,payload,scales)
            except BaseException:
                # Failed publication may have left a valid orphan or temporary
                # artifact. Never infer its absence from our expected events.
                self._dirty = True
                raise
        owner = _WriterLock(self.root)
        try:
            return self._put_locked(provenance,payload,scales)
        finally:
            owner.close()

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
            if p['sinq']['algorithm']=='identity' and not np.all(values==1):
                raise ValueError('identity metadata requires an actual identity compensation')
        desc = {'schema_version':1,'provenance':p,'payload':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload),'scales':None if scales is None else hashlib.sha256(scales).hexdigest(),'scale_bytes':0 if scales is None else len(scales)}
        key = digest(desc)
        identity=self.root/'identities'/f'{digest(p)}.json'
        if identity.exists() and identity.read_bytes()!=canonical_json({'candidate_id':key}):
            raise ValueError('identical candidate provenance has different native bytes')
        path = self.root/'tensors'/f'{key}.json'
        if path.exists():
            self.get(key)
            return key
        self._budget(len(payload)+(len(scales) if scales is not None else 0)+len(canonical_json(desc))+len(canonical_json({'candidate_id':key})))
        for h, data in ((desc['payload'],payload),(desc['scales'],scales)):
            if data is not None:
                self._publish(self.root/'blobs'/h,data)
        self._publish(identity,canonical_json({'candidate_id':key}))
        self._publish(path,canonical_json(desc))
        self.get(key)
        return key

    def get(self, key, *, expected=None):
        sha(key)
        self._sync()
        if self._depth and self._index is None:
            self._census()
            self._check_budget(0, 0)
        if self._index is not None and key in self._receipts:
            candidate = self._receipts[key]
            if expected is not None and any(candidate.provenance.get(k) != v for k,v in expected.items()):
                raise ValueError('candidate provenance mismatch')
            self.index_stats['receipt_hits'] += 1
            return Candidate(key, copy.deepcopy(candidate.provenance), candidate.payload, candidate.scales)
        self.index_stats['receipt_misses'] += 1
        path = self.root/'tensors'/f'{key}.json'
        self._guard(path)
        if path.stat().st_size > min(self.max_bytes, 1 << 20):
            raise ValueError('descriptor budget exceeded')
        desc = json.loads(path.read_bytes())
        if desc.get('schema_version') != 1 or digest(desc) != key:
            raise ValueError('candidate descriptor hash/schema mismatch')
        p = desc['provenance']
        size = validate_provenance(p)
        identity=self.root/'identities'/f'{digest(p)}.json'
        self._guard(identity)
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
            self._guard(blob)
            if blob.stat().st_size != count or self._hash(blob) != h:
                raise ValueError('candidate blob checksum/size mismatch')
            paths.append(blob)
        if paths[1] is not None:
            scales=np.memmap(paths[1],dtype='<f2',mode='r')
            try:
                if not _valid_scales(scales):
                    raise ValueError('invalid F16 companion')
                if p['sinq']['algorithm']=='identity' and not np.all(scales==1):
                    raise ValueError('identity metadata requires an actual identity compensation')
            finally:scales._mmap.close()
        candidate = Candidate(key,p,*paths)
        if self._index is not None:
            self._receipts[key] = Candidate(key,copy.deepcopy(p),*paths)
        return candidate

    def _guard(self, path):
        if self._index is not None:
            try:
                self._index.guard(path)
                self.index_stats['guarded_files'] = self._index.file_count
            except OSError as error:
                self._fallback(error)
