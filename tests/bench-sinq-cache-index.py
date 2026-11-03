"""Bounded synthetic CAS I/O only; no model throughput or quality claims."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import nullcontext

os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
_spec = importlib.util.spec_from_file_location('cache_fixture', Path(__file__).with_name('test-sinq-candidate-cache.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def measure(count, verified):
    with tempfile.TemporaryDirectory() as directory:
        store = _fixture.CandidateStore(directory, max_files=4 * count + 100, max_bytes=64 << 20)
        start = time.perf_counter()
        with store.verified_session() if verified else nullcontext():
            keys = [store.put(_fixture.provenance(f'blk.{i}.weight'), bytes(352), _fixture.np.ones(256, dtype='<f2').tobytes()) for i in range(count)]
            inserted = time.perf_counter()
            for _ in range(2):
                for key in keys:
                    store.get(key, expected={'qtype': 'Q5_0'})
            read = time.perf_counter()
            stats = dict(store.index_stats)
        return {'count': count, 'insert_seconds': inserted-start, 'read_twice_seconds': read-inserted, 'index_stats': stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 10000:
        parser.error('count must be between 1 and 10000')
    result = {'platform': os.name, 'scope': 'synthetic CAS only', 'default': measure(args.count, False), 'verified': measure(args.count, True)}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
