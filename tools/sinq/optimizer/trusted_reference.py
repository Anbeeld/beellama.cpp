"""Mandatory Windows file guards and explicitly historical large-reference trust.

No normal operation hashes reference probabilities. An inactive historical interval
is NOT authenticated by these guards; only an explicit audit establishes a fresh
content digest. Separate native readers use the canonical guarded path: ancestor
and file deny-delete handles prevent rebinding it for the lifetime of the session.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import struct

from .cache import _win_api
from .resources import file_digest


class GuardedFile:
    """Read-sharing only, exact single-link file object; never a stat hash cache."""
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.handles = []
        self.handle = None
        self.identity = None
        self._digest = None

    def __enter__(self):
        if os.name != 'nt':
            raise RuntimeError('immutable input guards require Windows; no stat-only fallback')
        self.api = _win_api()  # Same CreateFileW deny-write/delete semantics as CandidateStore.
        class Info(c.Structure):
            _fields_ = [('attributes', w.DWORD), ('created', w.FILETIME), ('accessed', w.FILETIME),
                        ('written', w.FILETIME), ('volume', w.DWORD), ('size_high', w.DWORD),
                        ('size_low', w.DWORD), ('links', w.DWORD), ('index_high', w.DWORD), ('index_low', w.DWORD)]
        class FileIdInfo(c.Structure):
            _fields_ = [('volume', c.c_uint64), ('file_id', c.c_ubyte * 16)]
        self.FileIdInfo = FileIdInfo
        self.api.GetFileInformationByHandleEx.argtypes = [w.HANDLE, c.c_int, w.LPVOID, w.DWORD]
        self.api.GetFileInformationByHandleEx.restype = w.BOOL
        self.Info = Info
        self.api.GetFileInformationByHandle.argtypes = [w.HANDLE, c.POINTER(Info)]
        self.api.GetFileInformationByHandle.restype = w.BOOL
        self.api.GetFinalPathNameByHandleW.argtypes = [w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD]
        self.api.GetFinalPathNameByHandleW.restype = w.DWORD
        try:
            self.api.GetDriveTypeW.argtypes = [w.LPCWSTR]
            self.api.GetDriveTypeW.restype = w.UINT
            self.api.GetVolumeInformationW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD,
                                                       c.POINTER(w.DWORD), c.POINTER(w.DWORD),
                                                       c.POINTER(w.DWORD), w.LPWSTR, w.DWORD]
            self.api.GetVolumeInformationW.restype = w.BOOL
            filesystem = c.create_unicode_buffer(64)
            if (self.api.GetDriveTypeW(self.path.anchor) != 3 or
                    not self.api.GetVolumeInformationW(self.path.anchor, None, 0, None, None,
                                                       None, filesystem, len(filesystem)) or
                    filesystem.value not in ('NTFS', 'ReFS')):
                raise OSError('immutable guards require a local fixed NTFS/ReFS volume')
            # Guard ancestors first; opening reparse points themselves closes resolve/open races.
            for path in reversed(self.path.parents):
                h = self._open(path, 0x80, 3, 0x02200000)
                if self._info(h).attributes & 0x400:
                    raise OSError('reparse-point ancestors cannot be guarded')
            self.handle = self._open(self.path, 0x80000000, 1, 0x00200080)
            self.identity = self._identity(self.handle)
            self.check()
            return self
        except BaseException:
            self.close()
            raise

    def _open(self, path, access, sharing, flags):
        handle = self.api.CreateFileW(str(path), access, sharing, None, 3, flags, None)
        if handle == self.api.invalid:
            raise OSError(c.get_last_error(), 'cannot acquire immutable input guard', str(path))
        self.handles.append(handle)
        return handle

    def _info(self, handle):
        info = self.Info()
        if not self.api.GetFileInformationByHandle(handle, c.byref(info)):
            raise OSError(c.get_last_error(), 'cannot identify guarded file object')
        return info

    def _identity(self, handle):
        info = self._info(handle)
        if info.attributes & (0x400 | 0x10) or info.links != 1:
            raise OSError('guarded input must be a regular, non-reparse, single-link file')
        buf = c.create_unicode_buffer(32768)
        count = self.api.GetFinalPathNameByHandleW(handle, buf, len(buf), 0)
        if not count or count >= len(buf):
            raise OSError('cannot establish canonical guarded path')
        canonical = buf.value
        if canonical.startswith('\\\\?\\'):
            canonical = canonical[4:]
        if canonical.casefold() != str(self.path).casefold():
            raise OSError('canonical path alias rejected')
        file_id = self.FileIdInfo()
        if not self.api.GetFileInformationByHandleEx(handle, 18, c.byref(file_id), c.sizeof(file_id)):
            raise OSError(c.get_last_error(), 'cannot establish full 128-bit file identity')
        return {'canonical_path': canonical, 'volume_serial': file_id.volume,
                'file_id': int.from_bytes(bytes(file_id.file_id), 'little'),
                'size': (info.size_high << 32) | info.size_low}

    def check(self):
        if self.handle is None:
            raise RuntimeError('immutable input guard is not active')
        if self._identity(self.handle) != self.identity:
            raise RuntimeError('guarded file identity changed')
        # Prove the separate path-based reader resolves to this same object.
        h = self.api.CreateFileW(str(self.path), 0x80000000, 1, None, 3, 0x00200080, None)
        if h == self.api.invalid:
            raise OSError(c.get_last_error(), 'cannot reopen guarded reader path')
        try:
            if self._identity(h) != self.identity:
                raise RuntimeError('reader path differs from guarded file object')
        finally:
            self.api.CloseHandle(h)
        return self.identity

    def verified_sha256(self, expected):
        self.check()
        if self._digest is None:
            self._digest = file_digest(self.path)
        if self._digest != expected:
            raise ValueError('guarded input content differs from expected SHA256')
        return self._digest

    def close(self):
        for handle in reversed(self.handles):
            self.api.CloseHandle(handle)
        self.handles.clear()
        self.handle = None
        self._digest = None

    def __exit__(self, *exc):
        self.close()


class TrustedLargeReference:
    """Historical evidence + small alignment checks, NOT fresh content verification."""
    def __init__(self, lock_path, *, evaluation_policy=None):
        self.lock_path = Path(lock_path)
        self.policy = evaluation_policy or {}
        self.guard = None

    def __enter__(self):
        if self.lock_path.stat().st_size > 1 << 20:
            raise ValueError('reference evidence lock exceeds small-metadata budget')
        lock_bytes = self.lock_path.read_bytes()
        self.lock = json.loads(lock_bytes)
        self.lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        if self.lock.get('status') != 'PASS' or self.lock.get('reference_format') != '_logitF1':
            raise ValueError('missing or invalid frozen reference evidence lock')
        for key in ('reference_sha256', 'final_test_token_stream_sha256'):
            value = self.lock.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value):
                raise ValueError('invalid historical digest/token identity in reference lock')
        if any(type(self.lock.get(key)) is not int or self.lock[key] <= 0 for key in
               ('reference_bytes', 'reference_context', 'reference_vocabulary', 'reference_input_tokens', 'scored_targets')):
            raise ValueError('invalid reference dimensions in evidence lock')
        self.guard = GuardedFile(self.lock['reference_path'])
        self.guard.__enter__()
        try:
            if self.guard.identity['size'] != self.lock['reference_bytes']:
                raise ValueError('trusted reference size mismatch; explicit audit required, no automatic rehash')
            count = self.lock['reference_input_tokens']
            ctx, vocab = self.lock['reference_context'], self.lock['reference_vocabulary']
            expected_size = 24 + 4 * count + (count - 1) * (8 + 2 * (vocab + vocab % 2))
            if expected_size != self.lock['reference_bytes'] or self.lock['scored_targets'] != count - 1:
                raise ValueError('inconsistent reference format dimensions')
            with self.guard.path.open('rb') as stream:
                header = stream.read(24)
                if header != struct.pack('<8sIIQ', b'_logitF1', ctx, vocab, count):
                    raise ValueError('trusted reference header mismatch')
                # Only the small token identity region is hashed, never probability rows.
                token_hash = hashlib.sha256()
                remaining = count * 4
                while remaining:
                    block = stream.read(min(remaining, 1 << 20))
                    if not block:
                        raise ValueError('truncated reference token identity')
                    token_hash.update(block)
                    remaining -= len(block)
                if token_hash.hexdigest() != self.lock['final_test_token_stream_sha256']:
                    raise ValueError('trusted reference token identity mismatch')
            self.check()
            return self
        except BaseException:
            self.guard.close()
            raise

    @property
    def path(self):
        return self.guard.path

    def check(self):
        return self.guard.check()

    def provenance(self):
        self.check()
        return {'trust_mode': 'previously_verified_immutable_local_file', 'fresh_content_hash': False,
                'hash_performed_in_this_session': False,
                'expected_previously_verified_sha256': self.lock['reference_sha256'],
                'original_evidence_lock': str(self.lock_path.resolve()),
                'original_evidence_lock_sha256': self.lock_sha256,
                'file_identity': dict(self.guard.identity), 'format': self.lock['reference_format'],
                'context': self.lock['reference_context'], 'vocabulary': self.lock['reference_vocabulary'],
                'input_tokens': self.lock['reference_input_tokens'], 'evaluation_policy': self.policy,
                'reference_alignment_bytes_read': 24 + 4 * self.lock['reference_input_tokens'],
                'reference_content_hashes_per_normal_run': 0,
                'reference_content_bytes_hashed_per_normal_run': 0, 'reference_sha256_new_bytes_read': 0,
                'historical_interval_note': 'No cryptographic proof of unchanged content while no guard was active.'}

    def __exit__(self, *exc):
        self.guard.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Explicit standalone reference content audit; never used by normal evaluation')
    parser.add_argument('command', choices=['audit-reference'])
    parser.add_argument('lock', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with TrustedLargeReference(args.lock) as reference:
        actual = reference.guard.verified_sha256(reference.lock['reference_sha256'])
        report = reference.provenance()
        report.update(trust_mode='freshly_audited_immutable_local_file',
                      fresh_content_hash=True, hash_performed_in_this_session=True,
                      audited_sha256=actual, reference_content_hashes_per_normal_run=None,
                      reference_content_bytes_hashed_per_normal_run=None, reference_sha256_new_bytes_read=reference.guard.identity['size'])
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
