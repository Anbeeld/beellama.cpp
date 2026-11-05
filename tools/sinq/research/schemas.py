"""Version-1 offline research configs. Validation reads no artifacts and creates nothing.

Paths are retained as supplied (relative to the invoking process). Artifact contents,
codec capabilities and physical output aliases must additionally be checked by the
operation before publication. Pairing consumes independently validated v1 activation
manifests; it does not replace the native activation loader's payload validation.
"""
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

FAMILIES = ('ordinary', 'awq_weight_activation', 'awq_activation_only')
NATIVE_QTYPES = frozenset(('q3_k', 'q4_k', 'iq1_s', 'iq1_m', 'iq2_xxs', 'iq2_xs',
    'iq2_s', 'iq3_xxs', 'iq3_s', 'iq4_xs', 'iq4_nl', 'q5_0', 'q5_1', 'q6_k', 'q8_0'))
# Single compatibility authority. Held-out is a partition policy, not a method.
METHOD_COMPATIBILITY = {
    family: frozenset(('local', 'blockwise', 'fisher_vjp_mc', 'groupwise_kl'))
    for family in (*FAMILIES, 'stock')
}
BUDGET_FIELDS = frozenset(('workspace_bytes', 'max_candidates', 'max_output_bytes',
    'max_files', 'max_tensors', 'max_tokens', 'max_probes', 'max_evaluations'))
COMMON = {'schema_version', 'operation', 'budgets', 'threads'}
OPERATION_FIELDS = {
    'generate': ({'source_gguf', 'activation_manifest', 'tensor_list', 'native_quantizer',
        'generators', 'alphas', 'correction_strengths', 'search', 'output_directory', 'native_bridge'},
        {'validation_manifest', 'pairing_manifest', 'stock_baseline_manifest', 'corpus_partitions_manifest'}),
    'score': ({'method', 'objective', 'candidate_manifest', 'capture_manifest', 'output_directory'},
        {'families', 'device', 'reference_dtype', 'cuda_devices', 'memory_policy', 'context', 'checkpoint', 'native_bridge', 'probes', 'seed',
         'validation_manifest', 'pairing_manifest', 'selection_policy', 'assignments', 'fisher_positions'}),
    'assemble': ({'source_gguf', 'candidate_manifest', 'assignment', 'output_gguf'}, {'native_loader', 'stock_baseline_manifest'}),
    'lock_corpora': ({'token_sources', 'partitions', 'output_manifest'}, set()),
    'lock_baseline': ({'source_gguf', 'stock_gguf', 'recipe', 'quantizer', 'native_loader', 'output_manifest'}, {'imatrix_path', 'timeout'}),
    'select': ({'source_gguf', 'candidate_manifest', 'baseline', 'groups', 'selection_manifest',
        'selection_split', 'evaluator', 'output_directory'}, {'max_passes', 'tolerance', 'stock_baseline_manifest', 'corpus_partitions_manifest'}),
    'validate': ({'artifact_manifest'}, {'source_gguf', 'native_loader'}),
}


def _fields(value, required, optional=(), label='config'):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    missing, unknown = set(required)-value.keys(), value.keys()-set(required)-set(optional)
    if missing or unknown:
        raise ValueError(f'{label}: missing {sorted(missing)}; unknown {sorted(unknown)}')


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{label} must be nonempty text')
    return value


def _integer(value, label, minimum=1):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError(f'{label} must be an integer in {minimum}..2^63-1')
    return value


def _number(value, label, low=0, high=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be finite in [{low},{high}]')
    return float(value) if value else 0.0


def _choice(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f'{label} must be one of {sorted(choices)}')
    return value


def _list(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f'{label} must be a nonempty list')
    return value


def _names(value, label):
    result = [_text(x, label) for x in _list(value, label)]
    if len(set(result)) != len(result):
        raise ValueError(f'{label} contains duplicates')
    return result


def _mapping(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{label} must be a nonempty mapping')
    for key, item in value.items():
        _text(key, label); _text(item, label)


def _candidate(family, alpha, strength):
    if family == 'ordinary' or strength == 0 or (family == 'awq_activation_only' and alpha == 0):
        family, alpha, strength = 'ordinary', 0.0, 0.0
    result = dict(family=family, alpha=float(alpha), **{'lambda': float(strength)})
    identity = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    result['stable_id'] = 'sinq-v1-' + hashlib.sha256(identity.encode()).hexdigest()
    return result


def _generate(c):
    names = _names(c['tensor_list'], 'tensor_list')
    q = c['native_quantizer']
    _fields(q, {'recipe', 'sinq_strip_width', 'sinq_iterations', 'sinq_strength'}, {'imatrix_path'}, 'native_quantizer')
    _mapping(q['recipe'], 'recipe')
    if set(q['recipe']) != set(names):
        raise ValueError('recipe must contain exactly tensor_list names')
    for qtype in q['recipe'].values():
        _choice(qtype, NATIVE_QTYPES, 'native qtype')
    _integer(q['sinq_strip_width'], 'sinq_strip_width')
    if q['sinq_strip_width'] not in (32, 64, 128, 256, 512):
        raise ValueError('unsupported sinq_strip_width')
    _integer(q['sinq_iterations'], 'sinq_iterations')
    if q['sinq_iterations'] not in (4,8,16):
        raise ValueError('unsupported sinq_iterations')
    q['sinq_strength'] = _number(q['sinq_strength'], 'sinq_strength')
    if 'imatrix_path' in q:
        _text(q['imatrix_path'], 'imatrix_path')
    if any(t in ('iq1_s', 'iq2_xxs', 'iq2_xs') for t in q['recipe'].values()) and 'imatrix_path' not in q:
        raise ValueError('recipe contains a native codec requiring imatrix_path')
    generators = _names(c['generators'], 'generators')
    for family in generators:
        _choice(family, FAMILIES, 'generator')
    alphas = sorted(set(_number(x, 'alpha') for x in _list(c['alphas'], 'alphas')))
    strengths = sorted(set(_number(x, 'correction strength') for x in _list(c['correction_strengths'], 'correction_strengths')))
    c['alphas'], c['correction_strengths'] = alphas, strengths
    c['generators'] = [f for f in FAMILIES if f in generators]
    s = c['search']
    _fields(s, {'strategy', 'local_objective', 'selection_policy', 'max_shortlist', 'refine_step', 'tolerance'}, label='search')
    _choice(s['strategy'], ('small_grid', 'legacy', 'coarse_then_refine'), 'strategy')
    _choice(s['local_objective'], ('output_l1', 'output_mse'), 'local_objective')
    _choice(s['selection_policy'], ('in_sample', 'held_out'), 'selection_policy')
    _integer(s['max_shortlist'], 'max_shortlist', 0)
    if s['strategy']=='coarse_then_refine' and s['max_shortlist']==0:
        raise ValueError('coarse_then_refine requires a nonzero shortlist')
    s['refine_step'] = _number(s['refine_step'], 'refine_step')
    if s['refine_step'] == 0:
        raise ValueError('refine_step must be positive')
    s['tolerance'] = _number(s['tolerance'], 'tolerance', high=1e300)
    if s['selection_policy'] == 'held_out' and not {'validation_manifest', 'pairing_manifest'} <= c.keys():
        raise ValueError('held_out requires validation_manifest and pairing_manifest')
    if ('validation_manifest' in c) != ('pairing_manifest' in c):
        raise ValueError('validation_manifest and pairing_manifest must appear together')
    candidates = [_candidate('ordinary', 0, 0)]
    if s['strategy'] == 'legacy':
        if s['local_objective'] != 'output_l1' or s['selection_policy'] != 'in_sample':
            raise ValueError('legacy requires in_sample output_l1')
        candidates += [_candidate('awq_weight_activation', i/20, 1) for i in range(20)]
    else:
        candidates += [_candidate(f, a, strength) for f in FAMILIES if f in generators
                       for a in alphas for strength in strengths]
    c['resolved_candidates'] = list({x['stable_id']: x for x in candidates}.values())
    count = len(c['resolved_candidates'])
    if s['strategy'] == 'coarse_then_refine':
        count += 2*s['max_shortlist']
    return len(names), count


def validate_execution_config(config, *, local=False):
    """Validate execution controls without importing Torch or touching CUDA."""
    c = copy.deepcopy(config)
    if local:
        if any(k in c for k in ('reference_dtype', 'cuda_devices', 'memory_policy')) or c.get('device', 'cpu') != 'cpu':
            raise ValueError('local scoring only accepts device cpu and no precision/placement/memory controls')
        c.setdefault('device', 'cpu')
        return c
    explicit = c.get('device')
    if explicit is not None and (not isinstance(explicit, str) or not re.fullmatch(r'cpu|cuda(?::(?:0|[1-9][0-9]*))?', explicit)):
        raise ValueError('device must be cpu, cuda, or cuda:<nonnegative index>')
    _choice(c.setdefault('reference_dtype', 'fp32'), ('fp32', 'bf16'), 'reference_dtype')
    if 'cuda_devices' in c:
        devices = _names(c['cuda_devices'], 'cuda_devices')
        if any(not re.fullmatch(r'cuda:(?:0|[1-9][0-9]*)', d) for d in devices):
            raise ValueError('cuda_devices requires absolute cuda:N entries')
        if explicit not in (None, 'cuda'):
            raise ValueError('cuda_devices conflicts with device')
        c['device'] = 'cuda'
    else:
        c.setdefault('device', 'cpu')
    if 'memory_policy' in c:
        m = c['memory_policy']
        _fields(m, (), ('host_limit_bytes', 'cuda_limit_bytes', 'reserve_bytes'), 'memory_policy')
        for key in ('host_limit_bytes', 'reserve_bytes'):
            if key in m: _integer(m[key], key, 0 if key == 'reserve_bytes' else 1)
        if 'cuda_limit_bytes' in m:
            limits = m['cuda_limit_bytes']
            if not isinstance(limits, dict) or not limits:
                raise ValueError('cuda_limit_bytes must be a nonempty mapping')
            for name, limit in limits.items():
                if not isinstance(name, str) or not re.fullmatch(r'cuda:(?:0|[1-9][0-9]*)', name):
                    raise ValueError('cuda_limit_bytes requires absolute cuda:N keys')
                _integer(limit, 'cuda limit')
            if c['device'] == 'cpu': raise ValueError('CPU policy cannot have CUDA limits')
            if 'cuda_devices' in c and not limits.keys() <= set(c['cuda_devices']):
                raise ValueError('CUDA limits name unselected devices')
    return c


def validate_config(config):
    """Return a detached canonical dict; never open/create operation inputs or outputs."""
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    operation = _choice(config.get('operation'), OPERATION_FIELDS, 'operation')
    required, optional = OPERATION_FIELDS[operation]
    _fields(config, (COMMON-{'threads'}) | required, {'threads'} | optional)
    c = copy.deepcopy(config)
    if type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    _integer(c.setdefault('threads', os.cpu_count() or 1), 'threads')
    if c['threads'] > 1024:
        raise ValueError('threads must not exceed 1024')
    _fields(c['budgets'], BUDGET_FIELDS, label='budgets')
    for key, value in c['budgets'].items():
        _integer(value, key)
    # Paths/text fields do not trigger filesystem I/O.
    for key in ('source_gguf', 'activation_manifest', 'validation_manifest', 'pairing_manifest',
                'output_directory', 'native_bridge', 'candidate_manifest', 'capture_manifest',
                'checkpoint', 'output_gguf', 'native_loader', 'selection_manifest', 'evaluator', 'artifact_manifest',
                'stock_baseline_manifest', 'stock_gguf', 'quantizer', 'output_manifest', 'imatrix_path', 'corpus_partitions_manifest'):
        if key in c:
            _text(c[key], key)
    tensors, candidates = 0, 0
    if operation == 'generate':
        tensors, candidates = _generate(c)
    elif operation == 'score':
        method = _choice(c['method'], ('local', 'blockwise', 'fisher_vjp_mc'), 'method')
        if 'fisher_positions' in c and method != 'fisher_vjp_mc':
            raise ValueError('fisher_positions is only valid for fisher_vjp_mc')
        objectives = ('fisher_quadratic',) if method == 'fisher_vjp_mc' else ('output_l1', 'output_mse')
        _choice(c['objective'], objectives, 'objective')
        for family in _names(c.setdefault('families', list(FAMILIES)), 'families'):
            if family not in METHOD_COMPATIBILITY or method not in METHOD_COMPATIBILITY[family]:
                raise ValueError('unsupported family/method combination')
        c.update(validate_execution_config(c, local=method == 'local'))
        _choice(c.setdefault('selection_policy', 'in_sample'), ('in_sample', 'held_out'), 'selection_policy')
        if method!='local' and c['selection_policy']=='held_out':
            raise ValueError('advanced held_out capture provenance is not supported; use local held_out scoring')
        if method!='local' and any(k in c for k in ('validation_manifest','pairing_manifest')):
            raise ValueError('advanced scoring cannot accept unused validation/pairing manifests')
        if c['selection_policy'] == 'held_out' and not {'validation_manifest', 'pairing_manifest'} <= c.keys():
            raise ValueError('held_out requires validation_manifest and pairing_manifest')
        if 'assignments' in c:
            if method=='local':raise ValueError('joint assignments require an advanced scoring method')
            if not isinstance(c['assignments'],dict) or not c['assignments']:
                raise ValueError('assignments must contain named nonempty tensor mappings')
            for name,assignment in c['assignments'].items():
                _text(name,'assignment name');_mapping(assignment,'assignment')
                for key in assignment.values():_hash(key,'candidate ID')
                if len(assignment)>c['budgets']['max_tensors']:raise ValueError('joint tensor budget exceeded')
            if len(c['assignments'])>c['budgets']['max_evaluations']:raise ValueError('joint observation budget exceeded')
        if method == 'local' and any(k in c for k in ('context', 'checkpoint', 'probes', 'seed')):
            raise ValueError('local scoring cannot accept blockwise/Fisher controls')
        if method != 'local':
            if not {'checkpoint', 'native_bridge', 'context'} <= c.keys():
                raise ValueError('advanced scoring requires checkpoint, native_bridge, context')
            _choice(c['context'], ('reference_inputs', 'selected_quantized_inputs'), 'context')
        if method == 'blockwise' and any(k in c for k in ('probes', 'seed')):
            raise ValueError('blockwise cannot accept Fisher controls')
        if method == 'fisher_vjp_mc':
            _integer(c.get('probes'), 'probes'); _integer(c.get('seed'), 'seed', 0)
            if c['probes'] > c['budgets']['max_probes']:
                raise ValueError('probe budget exceeded')
            if 'fisher_positions' in c:
                positions = c['fisher_positions']
                if not isinstance(positions, list) or not positions:
                    raise ValueError('fisher_positions must be a nonempty list of [batch,time] pairs')
                normalized = []
                for position in positions:
                    if (not isinstance(position, list) or len(position) != 2 or
                            any(type(index) is not int or index < 0 for index in position)):
                        raise ValueError('fisher_positions entries must be nonnegative integer [batch,time] pairs')
                    normalized.append(tuple(position))
                if len(set(normalized)) != len(normalized):
                    raise ValueError('fisher_positions must be unique')
                if 'max_probes' not in c['budgets'] or len(positions) * c['probes'] > c['budgets']['max_probes']:
                    raise ValueError('fisher_positions exceed the VJP budget')
    elif operation == 'lock_corpora':
        _mapping(c['token_sources'],'token_sources')
        _fields(c['partitions'], {'calibration','selection','final_test'}, {'validation'}, 'corpus partitions')
        for role,chunks in c['partitions'].items():
            if role=='validation' and chunks is None:continue
            for chunk in _list(chunks,'original token chunks'):
                _fields(chunk,{'source','start','end'},label='token chunk')
                _text(chunk['source'],'token source');_integer(chunk['start'],'start',0);_integer(chunk['end'],'end')
                if chunk['source'] not in c['token_sources'] or chunk['start']>=chunk['end']:raise ValueError('invalid token source range')
        if len(c['token_sources'])+1>c['budgets']['max_files']:raise ValueError('corpus file budget exceeded')
        if any(Path(p).resolve()==Path(c['output_manifest']).resolve() for p in c['token_sources'].values()):raise ValueError('corpus output aliases input')
    elif operation == 'lock_baseline':
        _mapping(c['recipe'],'complete stock recipe');tensors=len(c['recipe']);candidates=1
        for qtype in c['recipe'].values():
            if qtype!=qtype.upper():raise ValueError('stock recipe requires uppercase exact native qtypes')
        _integer(c.setdefault('timeout',3600),'timeout')
        if c['budgets']['max_files']<3:raise ValueError('stock lock needs three temporary/output files')
    elif operation == 'assemble':
        _mapping(c['assignment'], 'assignment'); tensors = len(c['assignment'])
    elif operation == 'select':
        _choice(c['selection_split'], ('selection',), 'selection_split (final-test optimization prohibited)')
        _mapping(c['baseline'], 'baseline'); tensors = len(c['baseline'])
        if not isinstance(c['groups'], dict) or not c['groups']:
            raise ValueError('groups must be a nonempty mapping')
        seen = set()
        for name, members in c['groups'].items():
            _text(name, 'group'); members = _names(members, 'group members')
            if seen.intersection(members) or not set(members) <= c['baseline'].keys():
                raise ValueError('overlapping or unmatched group members')
            seen.update(members)
        _integer(c.setdefault('max_passes', 1), 'max_passes')
        c['tolerance'] = _number(c.setdefault('tolerance', 0), 'tolerance', high=1e300)
    b = c['budgets']
    if tensors > b['max_tensors'] or candidates > b['max_candidates']:
        raise ValueError('tensor or candidate budget exceeded')
    # Conservative work bound; byte/token limits remain hard runtime caps, since
    # dimensions, actual payload sizes and token counts require artifact inspection.
    evaluations = tensors*candidates
    if evaluations > b['max_evaluations']:
        raise ValueError('evaluation budget exceeded')
    c.setdefault('resolved_candidates', [])
    c['resource_bounds'] = dict(tensor_count=tensors, candidates_per_tensor=candidates,
        max_candidate_evaluations=evaluations, **b)
    output = c.get('output_gguf', c.get('output_directory',c.get('output_manifest')))
    if output is not None:
        target = Path(output).resolve()
        for key in ('source_gguf', 'activation_manifest', 'validation_manifest', 'pairing_manifest',
                    'candidate_manifest', 'capture_manifest', 'selection_manifest', 'stock_baseline_manifest',
                    'stock_gguf', 'quantizer', 'imatrix_path', 'corpus_partitions_manifest'):
            if key in c and Path(c[key]).resolve() == target:
                raise ValueError('output aliases input ' + key)
    return c


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field: ' + key)
        result[key] = value
    return result


def load_config(path):
    """Read UTF-8 JSON with duplicate-field rejection, then validate without writes."""
    with Path(path).open(encoding='utf-8') as stream:
        return validate_config(json.load(stream, object_pairs_hook=_unique_object))


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
        raise ValueError(label + ' must be lowercase SHA256')
    return value


def validate_pairing(pairing, train_manifest, val_manifest):
    """Check split provenance in addition to independently validated activation v1s.

    Split chunks use half-open token offsets in the original token source. Training
    and validation chunks may touch but never intersect. row_ids are global source
    sample identities, not per-tensor row numbers. tiny_reservoir is test-only
    provenance and is retained explicitly, never relabeled as a production split.
    """
    _fields(pairing, {'schema_version', 'source_sha256', 'split_method', 'train', 'validation'}, label='pairing')
    if type(pairing['schema_version']) is not int or pairing['schema_version'] != 1:
        raise ValueError('pairing schema_version must be 1')
    _choice(pairing['split_method'], ('split_before_reservoir', 'tiny_reservoir'), 'split_method')
    source = _hash(pairing['source_sha256'], 'source_sha256')
    for manifest in (train_manifest, val_manifest):
        if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
            raise ValueError('activation manifest must be independently valid v1')
        if manifest.get('source_sha256') != source:
            raise ValueError('activation source identity mismatch')
    if any(train_manifest.get(k) != val_manifest.get(k) for k in ('architecture', 'source_revision')):
        raise ValueError('activation source identity mismatch')
    for key in ('train', 'validation'):
        split = pairing[key]
        _fields(split, {'split_id', 'token_sha256', 'chunks', 'row_ids'}, label=key)
        _text(split['split_id'], 'split_id'); _hash(split['token_sha256'], 'token_sha256')
        manifest = train_manifest if key == 'train' else val_manifest
        if split['token_sha256'] != manifest.get('sampling', {}).get('token_sha256'):
            raise ValueError('split token digest mismatch')
        _names(split['row_ids'], 'row_ids')
        previous = []
        for chunk in _list(split['chunks'], 'chunks'):
            _fields(chunk, {'token_source_sha256', 'start', 'end'}, label='chunk')
            _hash(chunk['token_source_sha256'], 'token_source_sha256')
            if chunk['token_source_sha256'] != manifest.get('sampling', {}).get('corpus_sha256'):
                raise ValueError('chunk token source does not match activation corpus provenance')
            _integer(chunk['start'], 'start', 0); _integer(chunk['end'], 'end')
            if chunk['end'] <= chunk['start']:
                raise ValueError('empty/reversed token chunk')
            for other in previous:
                if _overlap(chunk, other):
                    raise ValueError('duplicate/overlapping chunks within split')
            previous.append(chunk)
    train, val = pairing['train'], pairing['validation']
    if train['split_id'] == val['split_id'] or set(train['row_ids']).intersection(val['row_ids']):
        raise ValueError('training/validation split or row overlap')
    if any(_overlap(a,b) for a in train['chunks'] for b in val['chunks']):
        raise ValueError('training/validation token-source chunk overlap')
    return copy.deepcopy(pairing)


def _overlap(a, b):
    return (a['token_source_sha256'] == b['token_source_sha256'] and
            a['start'] < b['end'] and b['start'] < a['end'])
