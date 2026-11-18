"""Provenance closure for explicitly approved replacement stock controls."""
from pathlib import Path
from tools.sinq.optimizer.resources import digest
from tools.sinq.optimizer.schemas import read_json
from tools.sinq.optimizer.trusted_reference import GuardedFile


def validate_stock_baseline_control(model, expected_source_sha256):
    if model.get('kind')!='stock-baseline':raise ValueError('not an explicit stock-baseline replacement')
    path=Path(model['stock_baseline_manifest_path'])
    if path.stat().st_size>16<<20:raise ValueError('stock baseline manifest is not bounded metadata')
    with GuardedFile(path) as guard:
        guard.verified_sha256(model['stock_baseline_manifest_sha256'])
        baseline=read_json(path,max_bytes=16<<20)
        if (baseline.get('checksum')!=digest({k:v for k,v in baseline.items() if k!='checksum'}) or
                baseline.get('source_sha256')!=expected_source_sha256 or
                baseline.get('baseline_sha256')!=model['expected_sha256']):
            raise ValueError('replacement stock baseline provenance mismatch')
        guard.check()
