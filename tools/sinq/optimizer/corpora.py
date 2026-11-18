"""Locked original token occurrences, independent of capture serialization and paths.

Source and stage hashes cover raw little-endian uint32 token bytes. Chunks retain
explicit concatenation order; independence is disjoint original source indices,
not distinct token values. This module performs no model evaluation.
"""
import copy
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

from .resources import canonical_json, digest, reject_alias

ROLES = ('calibration', 'validation', 'selection', 'final_test')
_REQUIRED = frozenset(('calibration', 'selection', 'final_test'))
_BLOCK = 1 << 20
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _fields(value, fields, optional=()):
    if not isinstance(value, dict) or set(fields) - value.keys() or value.keys() - set(fields) - set(optional):
        raise ValueError('invalid corpus fields')


def _integer(value, minimum=1):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError('invalid corpus integer or resource bound')
    return value


def _hash(value):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError('invalid corpus SHA256')
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError('corpus source labels must be nonempty text')


def _check_overlaps(intervals):
    for ranges in intervals.values():
        previous_end = -1
        for start, end in sorted(ranges):
            if start < previous_end:
                raise ValueError('overlapping original corpus ranges within or across roles')
            previous_end = end


def _validate(manifest):
    _fields(manifest, {'schema_version', 'kind', 'sources', 'partitions', 'checksum'})
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 2:
        raise ValueError('unsupported corpus schema_version')
    if manifest['kind'] != 'sinq-corpus-partitions':
        raise ValueError('invalid corpus manifest kind')
    _hash(manifest['checksum'])
    if digest({k: v for k, v in manifest.items() if k != 'checksum'}) != manifest['checksum']:
        raise ValueError('corpus manifest checksum mismatch')
    sources = manifest['sources']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('corpus sources must be a nonempty object')
    for key, source in sources.items():
        _hash(key)
        _fields(source, {'token_count'})
        _integer(source['token_count'])
    _fields(manifest['partitions'], ROLES)
    intervals = {}
    for role in ROLES:
        stage = manifest['partitions'][role]
        if stage is None:
            if role in _REQUIRED:
                raise ValueError('calibration, selection and final_test partitions are required')
            continue
        _fields(stage, {'token_sha256', 'token_count', 'chunks'})
        _hash(stage['token_sha256'])
        _integer(stage['token_count'])
        if not isinstance(stage['chunks'], list) or not stage['chunks']:
            raise ValueError('nonempty corpus partition chunks required')
        count = 0
        for chunk in stage['chunks']:
            _fields(chunk, {'token_source_sha256', 'start', 'end'})
            key = _hash(chunk['token_source_sha256'])
            start, end = _integer(chunk['start'], 0), _integer(chunk['end'])
            if key not in sources or not start < end <= sources[key]['token_count']:
                raise ValueError('invalid original corpus range')
            intervals.setdefault(key, []).append((start, end))
            count += end-start
        if count != stage['token_count']:
            raise ValueError('corpus partition token_count mismatch')
    _check_overlaps(intervals)
    return copy.deepcopy(manifest)


def _source_count(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0 or info.st_size % 4:
        raise ValueError('token source must be nonempty regular raw little-endian uint32 data')
    return info.st_size // 4


def _source_digest(path, token_count):
    """Read a bounded stream, checking even unselected token IDs and truncation."""
    if _source_count(path) != token_count:
        raise ValueError('token source size changed')
    remaining = token_count * 4
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while remaining:
            block = handle.read(min(_BLOCK, remaining))
            if not block or len(block) % 4:
                raise ValueError('token source truncated or changed')
            if max(block[3::4], default=0) > 127:
                raise ValueError('token IDs must be in signed-token range 0..2^31-1')
            result.update(block)
            remaining -= len(block)
        if handle.read(1):
            raise ValueError('token source grew during hashing')
    return result.hexdigest()


def _stage_digest(chunks, paths):
    result = hashlib.sha256()
    for chunk in chunks:
        path = paths[chunk['token_source_sha256']]
        begin, stop = chunk['start']*4, chunk['end']*4
        with path.open('rb') as handle:
            if os.fstat(handle.fileno()).st_size < stop:
                raise ValueError('token source truncated before range mapping')
            while begin < stop:
                end = min(stop, begin+_BLOCK)
                offset = begin - begin % mmap.ALLOCATIONGRANULARITY
                # Map bounded windows, not the entire potentially large source.
                with mmap.mmap(handle.fileno(), end-offset, access=mmap.ACCESS_READ, offset=offset) as segment:
                    result.update(segment[begin-offset:end-offset])
                begin = end
    return result.hexdigest()


def _publish(output, data):
    """Exclusive atomic publication: even an identical existing output is forbidden."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending = tempfile.mkstemp(prefix='.pending-corpora-', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, output)
    finally:
        os.unlink(pending)


def lock_corpora(token_sources, partitions, output, *, max_tokens, max_output_bytes, max_files):
    """Hash local raw .u32 files and publish a canonical immutable partition lock.

    max_tokens bounds the sum of all input file token counts (including aliases),
    and max_files counts supplied input files plus the single manifest output.
    All validation and input rehashing precede publication. No source paths,
    caller labels or token values enter the persisted manifest.
    """
    for bound in (max_tokens, max_output_bytes, max_files):
        _integer(bound)
    if not isinstance(token_sources, dict) or not token_sources:
        raise ValueError('nonempty corpus token_sources mapping required')
    if len(token_sources)+1 > max_files:
        raise ValueError('corpus file budget exceeded')
    _fields(partitions, _REQUIRED, {'validation'})
    paths = {}
    counts = {}
    for label, path in token_sources.items():
        _text(label)
        paths[label] = Path(path)
        counts[label] = _source_count(paths[label])
    if sum(counts.values()) > max_tokens:
        raise ValueError('corpus input token budget exceeded')
    output = Path(output)
    reject_alias(output, paths.values())
    if output.exists() or output.is_symlink():
        raise FileExistsError('corpus output already exists')
    # Validate caller ranges before any token scan or mapping.
    for role, chunks in partitions.items():
        if chunks is None and role == 'validation':
            continue
        if not isinstance(chunks, list) or not chunks:
            raise ValueError('nonempty corpus role partitions required')
        for chunk in chunks:
            _fields(chunk, {'source', 'start', 'end'})
            _text(chunk['source'])
            start, end = _integer(chunk['start'], 0), _integer(chunk['end'])
            if chunk['source'] not in paths or not start < end <= counts[chunk['source']]:
                raise ValueError('invalid original corpus range')
    hashes = {label: _source_digest(path, counts[label]) for label, path in paths.items()}
    source_records = {hashes[label]: {'token_count': counts[label]} for label in paths}
    source_paths = {hashes[label]: path for label, path in paths.items()}
    normalized = {}
    intervals = {}
    selected_count = 0
    for role in ROLES:
        if partitions.get(role) is None:
            normalized[role] = None
            continue
        chunks = [dict(token_source_sha256=hashes[c['source']], start=c['start'], end=c['end'])
                  for c in partitions[role]]
        count = sum(c['end']-c['start'] for c in chunks)
        selected_count += count
        for chunk in chunks:
            intervals.setdefault(chunk['token_source_sha256'], []).append((chunk['start'], chunk['end']))
        normalized[role] = dict(token_count=count, chunks=chunks)
    _check_overlaps(intervals)
    if selected_count > max_tokens:
        raise ValueError('corpus partition token budget exceeded')
    for stage in normalized.values():
        if stage is not None:
            stage['token_sha256'] = _stage_digest(stage['chunks'], source_paths)
    manifest = dict(schema_version=2, kind='sinq-corpus-partitions', sources=source_records, partitions=normalized)
    manifest['checksum'] = digest(manifest)
    manifest = _validate(manifest)
    data = canonical_json(manifest)
    if len(data) > max_output_bytes:
        raise ValueError('corpus output byte budget exceeded')
    # Reopen every supplied path, including aliases, so substitution or edits
    # after the initial scan cannot silently publish stale range provenance.
    for label, path in paths.items():
        if _source_digest(path, counts[label]) != hashes[label]:
            raise ValueError('token source digest changed before publication')
    _publish(output, data)
    return manifest


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate corpus JSON field')
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError('nonfinite corpus JSON value')


def load_corpora(path):
    """Validate a v1 lock without requiring or opening the original token files."""
    with Path(path).open('r', encoding='utf-8') as handle:
        value = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return _validate(value)


def bind_corpora(manifest, bindings, selection_tokens=None):
    """Bind actual captured token digests, optionally checking native alignment.

    None means no corpus lock was supplied. Otherwise return a path-free identity
    with manifest_sha256 (the lock checksum), calibration and validation capture
    bindings. Capture manifest_sha256 identifies the capture, NOT this lock;
    corpus_sha256, if present, is serialization provenance, not token identity.
    Known counts must match; returned bindings always include the stage count.
    """
    if manifest is None:
        return None
    manifest = _validate(manifest)
    _fields(bindings, {'calibration', 'validation'})
    result = {'manifest_sha256': manifest['checksum']}
    for role in ('calibration', 'validation'):
        binding = bindings[role]
        stage = manifest['partitions'][role]
        if binding is None:
            result[role] = None
            continue
        if stage is None:
            raise ValueError('captured validation has no reserved corpus partition')
        _fields(binding, {'manifest_sha256', 'token_sha256'}, {'token_count', 'corpus_sha256'})
        _hash(binding['manifest_sha256'])
        _hash(binding['token_sha256'])
        if 'corpus_sha256' in binding:
            _hash(binding['corpus_sha256'])
        if binding['token_sha256'] != stage['token_sha256']:
            raise ValueError(role+' capture token digest does not match corpus partition')
        if 'token_count' in binding:
            _integer(binding['token_count'])
            if binding['token_count'] != stage['token_count']:
                raise ValueError(role+' capture token count does not match corpus partition')
        result[role] = dict(binding, token_count=stage['token_count'])
    if selection_tokens is not None:
        stage = manifest['partitions']['selection']
        if not isinstance(selection_tokens, list) or len(selection_tokens) != stage['token_count']:
            raise ValueError('native selection token count does not match corpus partition')
        token_hash = hashlib.sha256()
        for start in range(0, len(selection_tokens), _BLOCK//4):
            tokens = selection_tokens[start:start+_BLOCK//4]
            if any(type(token) is not int or not 0 <= token <= 2**31-1 for token in tokens):
                raise ValueError('native selection token IDs must be signed-range integers')
            token_hash.update(struct.pack('<'+'I'*len(tokens), *tokens))
        if token_hash.hexdigest() != stage['token_sha256']:
            raise ValueError('native selection token digest does not match corpus partition')
    return result
