"""Bounded file bridge to the canonical C++ producer encoder and GGML decoder.

No quantization formulas/codecs are implemented in Python. GGUF matrices are
[output,input], activation matrices [tokens,input], all packed bytes immutable.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def file_sha256(path):
    from .resources import file_digest
    return file_digest(path)


def _thread_count(threads=None):
    value = (os.cpu_count() or 1) if threads is None else threads
    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError('threads must be an integer in 1..1024')
    return value


def _run(binary, args, timeout=300, *, threads=None):
    threads = _thread_count(threads)
    env = {**os.environ, 'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = subprocess.run([str(Path(binary).resolve()), *map(str, args)], env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode:
        raise RuntimeError('native SINQ bridge failed: ' + result.stderr[-4000:])
    return json.loads(result.stdout)


def request(binary, config, directory, *, timeout=300):
    """Private temp request; filesystem outputs are managed by the native bridge."""
    config = dict(config)
    threads = _thread_count(config.get('threads'))
    config['threads'] = threads
    path = Path(directory) / 'request.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, sort_keys=True, allow_nan=False)
    return _run(binary, [path], timeout, threads=threads)


def resolve_plan(binary, config, inventory, *, threads=None):
    """Use the direct quantizer's native selector/parameter resolver, without encoding."""
    with tempfile.TemporaryDirectory(prefix='sinq-resolve-') as temp:
        direct=Path(temp)/'config.json';tensors=Path(temp)/'inventory.json'
        direct.write_text(json.dumps(config,sort_keys=True,allow_nan=False),encoding='utf-8')
        tensors.write_text(json.dumps(inventory,sort_keys=True,allow_nan=False),encoding='utf-8')
        return _run(binary,['--plan-config',direct,tensors],threads=threads)


def validate_activations(binary, manifest, source, *, threads=None):
    """Validate the independent activation format with bounded native workers."""
    threads=_thread_count(threads)
    return _run(binary, ['--validate-activations', Path(manifest).resolve(), Path(source).resolve(),threads], threads=threads)


def copy_tensor_bytes(tensor, destination, *, max_bytes):
    if tensor.n_bytes > max_bytes:
        raise ValueError('source tensor exceeds configured budget')
    raw = np.asarray(tensor.data).view(np.uint8).reshape(-1)
    if len(raw) != tensor.n_bytes:
        raise ValueError('source tensor byte cardinality mismatch')
    from .resources import observe_phase
    with observe_phase('native_source_export') as measured, Path(destination).open('xb') as stream:
        for offset in range(0, len(raw), 1024 * 1024):
            block=raw[offset:offset + 1024 * 1024]
            stream.write(block)
            if measured is not None:measured['bytes_written']+=block.nbytes


def decode(binary, candidate, *, first_row=0, rows=None, workspace_bytes=64 << 20, threads=None):
    """Decode a validated Candidate's indivisible native payload/scale pair.

    Caller must obtain Candidate through CandidateStore.get(). Output contains
    effective FP32 weights and is bounded to one explicit row range.
    """
    shape = candidate.provenance['shape']
    rows = shape[0] - first_row if rows is None else rows
    if type(first_row) is not int or type(rows) is not int or first_row < 0 or rows < 1 or first_row + rows > shape[0]:
        raise ValueError('decode row range mismatch')
    # Native bridge accepts matrices with >=2 rows; decode a one-row tail via a
    # two-row overlapping slab, returning only the requested row.
    native_first, native_rows = first_row, rows
    if rows == 1:
        if shape[0] < 2:
            raise ValueError('native decode requires a matrix with at least two rows')
        native_first = min(first_row, shape[0] - 2)
        native_rows = 2
    with tempfile.TemporaryDirectory(prefix='sinq-native-decode-') as d:
        output = Path(d) / 'decoded.f32'
        config = dict(schema_version=2, operation='decode', codec=candidate.provenance['qtype'].lower(),
                      rows=native_rows, columns=shape[1], first_row=native_first,
                      payload=str(candidate.payload.resolve()), output=str(output),
                      workspace_bytes=workspace_bytes, max_output_bytes=workspace_bytes,
                      threads=_thread_count(threads))
        if candidate.scales is not None:
            config['scales'] = str(candidate.scales.resolve())
        request(binary, config, d)
        out = np.fromfile(output, dtype='<f4').reshape(native_rows, shape[1])
        return out[first_row - native_first:first_row - native_first + rows].copy()


def rescore_many(binary, candidates, source_tensor, activation, *, objective='output_l1',
                 validation=None, workspace_bytes=64 << 20, threads=None, metrics=None):
    """Score ordered (ID, verified Candidate) pairs in one bounded tensor session.

    CandidateStore remains the payload/provenance trust boundary. Every candidate
    must bind the same freshly exported source. No result is returned unless the
    native response preserves the exact ID order. Optional metrics count fresh
    source I/O and the one native request, not whole-stage integrity work.
    """
    if objective not in ('output_l1', 'output_mse'):
        raise ValueError('unsupported local objective')
    candidates = list(candidates)
    ids = [key for key, _ in candidates]
    if (not ids or len(ids) > 1024 or any(not isinstance(key, str) or not key for key in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('invalid/duplicate candidate ID list')
    shape = list(map(int, source_tensor.shape))[::-1]
    expected = candidates[0][1].provenance['input_sha256']
    codec = candidates[0][1].provenance['qtype'].lower()
    items = []
    for key, candidate in candidates:
        if candidate.provenance['shape'] != shape:
            raise ValueError('rescore source shape mismatch')
        if candidate.provenance['input_sha256'] != expected:
            raise ValueError('rescore source reference mismatch')
        if candidate.provenance['qtype'].lower() != codec:
            raise ValueError('rescore codec mismatch')
        item = dict(candidate_id=key, payload=str(candidate.payload.resolve()))
        if candidate.scales is not None:
            item['scales'] = str(candidate.scales.resolve())
        items.append(item)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='sinq-native-score-many-') as d:
        weights = Path(d) / 'weights.bin'
        copy_tensor_bytes(source_tensor, weights, max_bytes=workspace_bytes)
        if file_sha256(weights) != expected:
            raise ValueError('rescore source bytes differ from candidate reference')
        config = dict(schema_version=2, operation='score-many', codec=codec,
                      source_type=source_tensor.tensor_type.name.lower(), rows=shape[0], columns=shape[1],
                      weights=str(weights), score_items=items,
                      activations=str(Path(activation['data_file']).resolve()), samples=activation['count'],
                      objective=objective, workspace_bytes=workspace_bytes, max_output_bytes=workspace_bytes,
                      output=str(Path(d) / 'score.json'), threads=_thread_count(threads))
        if validation is not None:
            config.update(held_out=True, validation=str(Path(validation['data_file']).resolve()),
                          validation_samples=validation['count'])
        result = request(binary, config, d)
        scores = result.get('scores', [])
        if ([item.get('candidate_id') for item in scores] != ids
                or any(not isinstance(item.get('score'), dict) for item in scores)):
            raise ValueError('native score-many candidate ID mapping mismatch')
    if metrics is not None:
        metrics.update(source_bytes_copied=int(source_tensor.n_bytes),
                       source_fresh_hash_bytes=int(source_tensor.n_bytes), process_launches=1,
                       native_compute_seconds=result['native_compute_seconds'],
                       native_scoring_seconds_inclusive=result['native_scoring_seconds_inclusive'],
                       source_bytes_read=result['source_bytes_read'],
                       activation_bytes_read=result['activation_bytes_read'],
                       elapsed_seconds=time.perf_counter() - start)
    return [(item['candidate_id'], item['score']) for item in scores]


def rescore(binary, candidate, source_tensor, activation, *, objective='output_l1',
            validation=None, workspace_bytes=64 << 20, threads=None):
    """Rescore immutable bytes without any native re-encoding (encode_count=0)."""
    if objective not in ('output_l1', 'output_mse'):
        raise ValueError('unsupported local objective')
    shape = candidate.provenance['shape']
    if list(map(int, source_tensor.shape))[::-1] != shape:
        raise ValueError('rescore source shape mismatch')
    with tempfile.TemporaryDirectory(prefix='sinq-native-score-') as d:
        weights = Path(d) / 'weights.bin'
        copy_tensor_bytes(source_tensor, weights, max_bytes=workspace_bytes)
        if file_sha256(weights) != candidate.provenance['input_sha256']:
            raise ValueError('rescore source bytes differ from candidate reference')
        config = dict(schema_version=2, operation='score', codec=candidate.provenance['qtype'].lower(),
                      source_type=source_tensor.tensor_type.name.lower(), rows=shape[0], columns=shape[1],
                      weights=str(weights), payload=str(candidate.payload.resolve()),
                      activations=str(Path(activation['data_file']).resolve()), samples=activation['count'],
                      objective=objective, workspace_bytes=workspace_bytes, max_output_bytes=workspace_bytes,
                      output=str(Path(d) / 'score.json'), threads=_thread_count(threads))
        if candidate.scales is not None:
            config['scales'] = str(candidate.scales.resolve())
        if validation is not None:
            config.update(held_out=True, validation=str(Path(validation['data_file']).resolve()),
                          validation_samples=validation['count'])
        return request(binary, config, d)
