"""Bounded deterministic artifact I/O. No network or implicit device selection."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Lock, get_ident
from time import perf_counter

_PERFORMANCE_OBSERVATIONS=ContextVar('sinq_performance_observations',default=None)
_PHASE_STACK=ContextVar('sinq_performance_phase_stack',default=())


@contextmanager
def observe_performance():
    """Opt-in invocation-local metrics; worker contexts share a locked aggregate."""
    metrics={};token=_PERFORMANCE_OBSERVATIONS.set((metrics,Lock()))
    try:yield metrics
    finally:_PERFORMANCE_OBSERVATIONS.reset(token)


@contextmanager
def observe_phase(name):
    """Inclusive and exclusive per-thread wall time; never sum concurrent workers.

    seconds retains the historical inclusive metric. exclusive_seconds subtracts
    same-thread nested phases only. Worker-thread totals are thread wall-seconds,
    not an additive attribution of the invocation's elapsed wall time.
    """
    state=_PERFORMANCE_OBSERVATIONS.get()
    if state is None:
        yield None
        return
    measured={'bytes_hashed':0,'bytes_read':0,'bytes_written':0};started=perf_counter()
    stack=_PHASE_STACK.get(); frame={'thread':get_ident(),'children':0.0,'state':state}
    token=_PHASE_STACK.set((*stack,frame))
    try:yield measured
    finally:
        elapsed=perf_counter()-started
        _PHASE_STACK.reset(token)
        if stack and stack[-1]['thread']==frame['thread'] and stack[-1]['state'] is state:
            stack[-1]['children']+=elapsed
        metrics,lock=state
        with lock:
            total=metrics.setdefault(name,dict(calls=0,seconds=0.0,exclusive_seconds=0.0,
                inclusive=True,time_scope='per-thread wall seconds; concurrent workers not additive',bytes_hashed=0,bytes_read=0,bytes_written=0))
            total['calls']+=1
            total['seconds']+=elapsed
            total['exclusive_seconds']+=max(0.0,elapsed-frame['children'])
            for counter in ('bytes_hashed','bytes_read','bytes_written'):total[counter]+=measured[counter]
            if name=='file_hash' and 'path' in measured:
                entry=total.setdefault('files',{}).setdefault(measured['path'],dict(calls=0,seconds=0.0,bytes_hashed=0,bytes_read=0))
                entry['calls']+=1;entry['seconds']+=elapsed
                for counter in ('bytes_hashed','bytes_read'):entry[counter]+=measured[counter]

_CACHE_OBSERVATIONS=ContextVar('sinq_cache_observations',default=None)


@contextmanager
def observe_caches():
    counters={};token=_CACHE_OBSERVATIONS.set(counters)
    try:yield counters
    finally:_CACHE_OBSERVATIONS.reset(token)


def record_cache(kind,hit):
    counters=_CACHE_OBSERVATIONS.get()
    if counters is not None:
        count=counters.setdefault(kind,dict(hits=0,misses=0))
        count['hits' if hit else 'misses']+=1


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path, chunk_bytes=1 << 20):
    # This distribution has an explicit historical-trust contract. Even legacy
    # utility scripts, failed preflights and broad inventories must fail closed,
    # never turn a cache miss into a 147.6 GB cryptographic scan. The standalone
    # audit-reference command deliberately uses its own opt-in streaming code.
    path=Path(path)
    if (path.name.casefold()=='qwen3.5-9b-wikitext2-64k-bf16.logits' or
            (path.suffix.casefold() in ('.logits','.bin') and path.stat().st_size>=147601497864)):
        raise ValueError('large reference hashing forbidden in normal operation; use TrustedLargeReference or explicitly invoke audit-reference')
    with observe_phase('file_hash') as measured:
        if measured is not None:measured['path']=str(path.resolve())
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while block := f.read(chunk_bytes):
                h.update(block)
                if measured is not None:
                    measured['bytes_hashed']+=len(block)
                    measured['bytes_read']+=len(block)
        return h.hexdigest()


def file_digests(paths, *, threads=1):
    """Hash independent files concurrently; each value remains an ordinary SHA-256."""
    from concurrent.futures import ThreadPoolExecutor
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('hash threads must be a positive integer')
    items=list(paths.items())
    def hash_item(item):
        name,path=item
        return name,file_digest(path)
    if threads==1 or len(items)<2:return dict(map(hash_item,items))
    context=copy_context()
    with ThreadPoolExecutor(max_workers=min(threads,len(items))) as pool:
        return dict(pool.map(lambda item:context.copy().run(hash_item,item),items))


def tensor_hashes(items, *, threads=1):
    """Hash independent (name, ndarray) buffers without copying or reordering them.

    Workers only read buffers; the caller must keep mappings open until return.
    The pool is joined on errors too, before callers can close those mappings.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('hash threads must be a positive integer')
    items = list(items)
    def hash_item(item):
        name, data = item
        with observe_phase('tensor_hash') as measured:
            raw = data.view(np.uint8).reshape(-1)
            h = hashlib.sha256()
            for start in range(0, raw.size, 1 << 20):
                block=raw[start:start + (1 << 20)]
                h.update(block)
                if measured is not None:measured['bytes_hashed']+=block.nbytes
            return name, h.hexdigest()
    if threads == 1 or len(items) < 2:
        return dict(map(hash_item, items))
    context=copy_context()
    with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
        return dict(pool.map(lambda item:context.copy().run(hash_item,item), items))


def reject_alias(output, inputs):
    output = Path(output)
    for item in inputs:
        item = Path(item)
        if output.resolve() == item.resolve() or (output.exists() and item.exists() and os.path.samefile(output, item)):
            raise ValueError('output aliases an input artifact')


def atomic_write(path, data, *, max_bytes, immutable=True):
    path = Path(path)
    if len(data) > max_bytes:
        raise ValueError('artifact byte budget exceeded')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with observe_phase('artifact_write') as measured, os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            if measured is not None:measured['bytes_written']+=len(data)
        if immutable:
            try:
                os.link(name, path)
            except FileExistsError:
                if path.stat().st_size != len(data) or file_digest(path) != hashlib.sha256(data).hexdigest():
                    raise ValueError('immutable artifact collision')
        else:
            os.replace(name, path)
        if file_digest(path) != hashlib.sha256(data).hexdigest():
            raise ValueError('artifact readback checksum mismatch')
    finally:
        if os.path.exists(name):
            os.unlink(name)


def peak_rss_bytes(pid=None):
    """Process high-water working set; optional child PID is Windows-only."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),
                *[(name,ctypes.c_size_t) for name in ('peak','working','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','pagefile_peak')]]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        info=Counters();info.cb=ctypes.sizeof(info)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x0400|0x0010,False,pid)
        if not handle:raise OSError(ctypes.get_last_error(),'cannot open process for peak memory')
        try:
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(info),info.cb):
                raise OSError(ctypes.get_last_error(),'cannot read process peak memory')
            return int(info.peak)
        finally:
            if pid is not None:kernel.CloseHandle(handle)
    if pid is not None:raise NotImplementedError('child peak working set is Windows-only')
    import resource
    import sys
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)
