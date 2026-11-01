"""Inspect actual GGUF storage, SINQ companions and matched tensor coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType

ROLES = {'attn_gate', 'attn_k', 'attn_output', 'attn_q', 'attn_qkv', 'attn_v',
         'ffn_down', 'ffn_gate', 'ffn_up', 'ssm_alpha', 'ssm_beta', 'ssm_out'}


SINQ_FIELDS = {'beellama.qk_sinq.' + name for name in
               ('version', 'tensors', 'algorithm', 'strip_width', 'iterations', 'producer', 'numerics')}


def validate_sinq_metadata(reader, path):
    for key in reader.fields:
        if key.startswith('beellama.qk_sinq.') and key not in SINQ_FIELDS:
            raise ValueError(f'{path}: unsupported SINQ metadata {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--practical', action='store_true', help='Allow different excluded-tensor dtypes')
    parser.add_argument('models', nargs='+', type=Path)
    parser.add_argument("--generic", action="store_true", help="Validate small graph fixtures without the pinned 427/248 Qwen policy")
    args = parser.parse_args()
    source = GGUFReader(args.source)
    validate_sinq_metadata(source, args.source)
    original = {t.name: t for t in source.tensors}
    eligible = set()
    for name in original:
        match = re.fullmatch(r'blk\.(\d+)\.([^.]+)\.weight', name)
        if match and int(match[1]) < 32 and match[2] in ROLES:
            eligible.add(name)
    if not args.generic and (len(original) != 427 or len(eligible) != 248):
        raise ValueError('This inventory policy requires the pinned Qwen3.5-9B text backbone')
    parameters = sum(int(t.n_elements) for t in original.values())
    eligible_parameters = sum(int(original[n].n_elements) for n in eligible)
    results = []
    for path in args.models:
        reader = GGUFReader(path)
        validate_sinq_metadata(reader, path)
        tensors = {t.name: t for t in reader.tensors}
        if len(tensors) != len(reader.tensors):
            raise ValueError(f'{path}: duplicate names')
        version = reader.fields.get('beellama.qk_sinq.version')
        declared_field = reader.fields.get('beellama.qk_sinq.tensors')
        declared = declared_field.contents() if declared_field is not None else []
        if not isinstance(declared,list) or len(declared) != len(set(declared)):
            raise ValueError(f'{path}: duplicate or invalid transformation manifest')
        declared = set(declared)
        if declared_field is not None and declared_field.types != [GGUFValueType.ARRAY, GGUFValueType.STRING]:
            raise ValueError(f'{path}: transformation manifest must be a string array')
        if declared and (version is None or version.types != [GGUFValueType.UINT32] or int(version.contents()) != 1):
            raise ValueError(f'{path}: missing or unsupported qk_sinq version')
        if version is not None and not declared:
            raise ValueError(f'{path}: version requires nonempty transformation manifest')
        if not declared.issubset(eligible):
            raise ValueError(f'{path}: transform outside supported primary-trunk mask')
        by_type = Counter()
        details = []
        companions = set()
        eligible_bytes = auxiliary_bytes = unquantized_bytes = 0
        for name, base in original.items():
            if name not in tensors:
                raise ValueError(f'{path}: missing {name}')
            tensor = tensors[name]
            if tuple(tensor.shape) != tuple(base.shape):
                raise ValueError(f'{path}: shape mismatch {name}')
            kind = GGMLQuantizationType(tensor.tensor_type).name
            by_type[kind] += int(tensor.n_elements)
            if not args.practical and name not in eligible and tensor.tensor_type != base.tensor_type:
                raise ValueError(f'{path}: changed exclusion {name}')
            if name in eligible:
                eligible_bytes += tensor.n_bytes
            if kind in ('F32', 'F16', 'BF16'):
                unquantized_bytes += tensor.n_bytes
            side_summary = None
            if name in declared:
                if kind not in ('Q3_K','Q4_K'):
                    raise ValueError(f'{path}: transformed base is not ordinary Q3_K/Q4_K')
                side_name = name + '.sinq_scale2'
                side = tensors.get(side_name)
                if side is None or GGMLQuantizationType(side.tensor_type).name != 'F16' or tuple(side.shape) != (int(tensor.shape[0]),):
                    raise ValueError(f'{path}: invalid companion {side_name}')
                values = side.data.reshape(-1)
                if not np.all(np.isfinite(values) & (values > 0)):
                    raise ValueError(f'{path}: nonpositive or nonfinite T')
                side_summary = {'min':float(values.min()),'max':float(values.max()),
                                'sha256':hashlib.sha256(side.data.tobytes()).hexdigest()}
                companions.add(side_name)
                auxiliary_bytes += side.n_bytes
            details.append({'name': name, 'shape': [int(v) for v in tensor.shape], 'dtype': kind,
                            'elements': int(tensor.n_elements), 'bytes': int(tensor.n_bytes), 'eligible': name in eligible, 'transformed':name in declared,
                            'payload_sha256':hashlib.sha256(tensor.data.tobytes()).hexdigest() if name in eligible else None,
                            'companion':side_summary})
        if set(tensors) != set(original) | companions:
            raise ValueError(f'{path}: unexpected or orphan tensors')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        record = {'path': str(path), 'sha256': digest, 'file_bytes': path.stat().st_size,
                  'base_parameters': parameters, 'whole_file_bpw': path.stat().st_size * 8 / parameters,
                  'eligible_parameters': eligible_parameters, 'eligible_tensors': len(eligible),
                  'eligible_weight_bpw_including_T': (eligible_bytes + auxiliary_bytes) * 8 / eligible_parameters,
                  'eligible_weight_bytes': int(eligible_bytes), 'auxiliary_bytes': int(auxiliary_bytes),
                  'unquantized_base_bytes': int(unquantized_bytes),
                  'transformed_tensors':len(declared), 'header_alignment_padding_bytes': int(path.stat().st_size - sum(t.n_bytes for t in tensors.values())),
                  'parameters_by_dtype': dict(by_type),
                  'parameter_percent_by_dtype': {k: v * 100 / parameters for k, v in by_type.items()},
                  'companions': len(companions), 'tensor_inventory': details}
        results.append(record)
        print(path.name, 'BPW', record['whole_file_bpw'], 'companions', len(companions), 'PASS')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
