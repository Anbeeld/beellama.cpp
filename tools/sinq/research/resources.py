"""Bounded deterministic artifact I/O. No network or implicit device selection."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path, chunk_bytes=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


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
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
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


def peak_rss_bytes():
    """Current process high-water working set; diagnostic only, not a budget."""
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
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
            raise OSError(ctypes.get_last_error(),'cannot read process peak memory')
        return int(info.peak)
    import resource
    import sys
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)
