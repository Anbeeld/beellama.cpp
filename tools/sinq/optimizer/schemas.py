"""Strict v2 stage schemas. No artifact reads, implicit search grids or plugins."""
import copy
import itertools
import json
import math
import os
import re
from pathlib import Path

NATIVE_QTYPES=frozenset(('Q3_K','Q4_K','IQ1_S','IQ1_M','IQ2_XXS','IQ2_XS','IQ2_S',
    'IQ3_XXS','IQ3_S','IQ4_XS','IQ4_NL','Q5_0','Q5_1','Q6_K','Q8_0'))
STAGES=('plan','generate','score','select','assemble','validate')
SCORERS={'local_output_l1':('local','output_l1'),'local_output_mse':('local','output_mse'),
    'blockwise_output_l1':('blockwise','output_l1'),'blockwise_output_mse':('blockwise','output_mse'),
    'fisher_vjp_mc':('fisher_vjp_mc','fisher_quadratic'),'model_kl':('groupwise_kl','kl')}


def _fields(value,required=(),optional=(),label='config'):
    if not isinstance(value,dict):raise ValueError(label+' must be an object')
    missing=set(required)-value.keys();unknown=value.keys()-set(required)-set(optional)
    if missing or unknown:raise ValueError(f'{label}: missing {sorted(missing)}; unknown {sorted(unknown)}')


def _text(v,label):
    if not isinstance(v,str) or not v.strip() or '\x00' in v:raise ValueError(label+' must be nonempty text')
    return v


def _integer(v,label,minimum=1):
    if type(v) is not int or not minimum<=v<=2**63-1:raise ValueError(label+' must be a bounded integer')
    return v


def _number(v,label,low=0,high=1):
    if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:raise ValueError(label+' must be finite in range')
    return float(v) if v else 0.0


def _choice(v,choices,label):
    if not isinstance(v,str) or v not in choices:raise ValueError(label+' must be one of '+str(sorted(choices)))
    return v


def _list(v,label):
    if not isinstance(v,list) or not v:raise ValueError(label+' must be a nonempty list')
    return v


def _names(v,label):
    result=[_text(x,label) for x in _list(v,label)]
    if len(set(result))!=len(result):raise ValueError(label+' contains duplicates')
    return result


def _mapping(v,label):
    if not isinstance(v,dict) or not v:raise ValueError(label+' must be a nonempty mapping')
    for k,x in v.items():_text(k,label);_text(x,label)
    return v


def parameters(value,*,partial=False):
    required=() if partial else ('generator','strip_width','iterations','strength')
    _fields(value,required,('generator','strip_width','iterations','strength','correction','alpha','lambda'),'parameters')
    p=copy.deepcopy(value)
    if 'generator' in p:_choice(p['generator'],('sinq','awq'),'generator')
    for key,allowed in [('strip_width',(8,16,32,64,128,256,512)),('iterations',(4,8,16))]:
        if key in p and (type(p[key]) is not int or p[key] not in allowed):raise ValueError('unsupported '+key)
    for key in ('strength','alpha','lambda'):
        if key in p:p[key]=_number(p[key],key)
    if p.get('generator')=='awq':
        if not {'correction','alpha','lambda'}<=p.keys():raise ValueError('AWQ requires correction, alpha and lambda')
        _choice(p['correction'],('weight_activation','activation_only'),'correction')
    elif any(k in p for k in ('correction','alpha','lambda')):raise ValueError('AWQ-only parameters on ordinary SINQ')
    return p


def expand_generators(generators,maximum):
    result={}
    for g in _list(generators,'candidate_generators'):
        _fields(g,('type',),('expansion','parameters','configurations'),'candidate generator')
        name=_choice(g['type'],('sinq','awq'),'generator type')
        if 'configurations' in g:
            if 'parameters' in g or 'expansion' in g:raise ValueError('choose explicit configurations or cartesian parameters')
            configs=_list(g['configurations'],'configurations')
        else:
            if g.get('expansion')!='cartesian':raise ValueError('search-space expansion must explicitly be cartesian')
            keys=('strip_width','iterations','strength')+(('correction','alpha','lambda') if name=='awq' else ())
            _fields(g.get('parameters'),keys,label='generator parameters')
            arrays=[_list(g['parameters'][k],k) for k in keys]
            count=math.prod(len(a) for a in arrays)
            if count+1>maximum:raise ValueError('candidate expansion including stock exceeds max_candidates_per_tensor')
            configs=[dict(zip(keys,v)) for v in itertools.product(*arrays)]
        for p in configs:
            p=parameters(dict(p,generator=name));key=json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False)
            result[key]=p
            if len(result)+1>maximum:raise ValueError('candidate expansion including stock exceeds max_candidates_per_tensor')
    return [result[k] for k in sorted(result)]


def _unique_object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON field: '+k)
        result[k]=v
    return result


def read_json(path,*,max_bytes=8<<20):
    p=Path(path)
    if p.stat().st_size>max_bytes:raise ValueError('JSON byte budget exceeded')
    return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=_unique_object,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON number: '+x)))


def load_config(path,stage):return validate_config(read_json(path),stage)


def validate_config(value,stage):
    if stage not in STAGES:raise ValueError('unknown optimizer subcommand')
    source=copy.deepcopy(value)
    if type(source.get('schema_version')) is not int or source['schema_version']!=2:raise ValueError('unsupported config schema version; expected 2')
    actual='generate' if stage=='plan' else stage
    groups={
        'generate':(('quantization','targets','candidate_generators'),('parameters',)),
        'score':(('scoring',),()),'select':(('selection',),()),
        'assemble':(('selection',),()),'validate':((),())}
    req,opt=groups[actual]
    _fields(source,('schema_version','inputs','limits')+req,('execution','output')+opt)
    c={'schema_version':2,'operation':actual}
    input_required={
        'generate':('source_gguf','native_bridge'),
        'score':('candidate_manifest','capture_manifest','native_bridge'),
        'select':('source_gguf','candidate_manifest'),
        'assemble':('source_gguf','candidate_manifest','native_loader'),
        'validate':('artifact_manifest',)}[actual]
    optional={'generate':('activation_manifest','validation_manifest','pairing_manifest','stock_baseline_manifest','corpus_partitions_manifest'),
        'score':('checkpoint','validation_manifest','pairing_manifest','stock_baseline_manifest'),
        'select':('selection_manifest','evaluator','stock_baseline_manifest','corpus_partitions_manifest','native_loader'),
        'assemble':('stock_baseline_manifest',),'validate':('source_gguf','native_loader','stock_baseline_manifest')}[actual]
    _fields(source['inputs'],input_required,optional,'inputs')
    for k,v in source['inputs'].items():c[k]=_text(v,k)
    execution=source.get('execution',{})
    _fields(execution,(),('threads','scoring'),'execution')
    c['threads']=_integer(execution.get('threads',os.cpu_count() or 1),'threads')
    if c['threads']>1024:raise ValueError('threads exceeds 1024')
    policy=execution.get('scoring',{})
    if actual!='score' and policy:raise ValueError('execution.scoring applies only to score stages')
    _fields(policy,(),('device','reference_dtype','cuda_devices','memory_policy'),'execution.scoring')
    c.update(policy)
    # Device strings and explicit placement use the preserved execution-policy validator.
    if 'reference_dtype' in c:_choice(c['reference_dtype'],('fp32','bf16'),'reference_dtype')
    limits=source['limits']
    shared={'workspace_bytes','max_artifact_bytes','max_files','max_tensors'}
    stage_limits={'generate':{'max_candidates_per_tensor'},'score':{'max_local_scores','max_block_scores','max_fisher_vjps','max_tokens'},
        'select':{'max_model_evaluations','max_tokens'},'assemble':set(),'validate':set()}[actual]
    _fields(limits,('workspace_bytes','max_artifact_bytes','max_files'),shared|stage_limits,'limits')
    for k,v in limits.items():_integer(v,k)
    b=dict(workspace_bytes=limits['workspace_bytes'],max_output_bytes=limits['max_artifact_bytes'],max_files=limits['max_files'],
        max_tensors=limits.get('max_tensors',100000),max_candidates=limits.get('max_candidates_per_tensor',128 if actual=='generate' else 1024))
    # Internal engine counters are not public config fields; only applicable limits enter identity.
    if actual=='generate':
        if b['workspace_bytes']>1<<30:raise ValueError('native generation workspace exceeds limit 1 GiB')
        if b['max_candidates']>1024:raise ValueError('max_candidates_per_tensor exceeds native limit 1024')
        b['max_evaluations']=b['max_candidates']*b['max_tensors']
    if actual=='score':
        scoring=source['scoring'];_fields(scoring,('kind',),('context','probes','seed','fisher_positions','assignments','selection_policy'),'scoring')
        kind=_choice(scoring['kind'],set(SCORERS)-{'model_kl'},'scorer');c['scorer_name']=kind
        c['method'],c['objective']=SCORERS[kind]
        if c['method']=='local' and b['workspace_bytes']>1<<30:raise ValueError('native local scoring workspace exceeds limit 1 GiB')
        c.update({k:v for k,v in scoring.items() if k!='kind'})
        c.setdefault('selection_policy','in_sample');_choice(c['selection_policy'],('in_sample','held_out'),'selection policy')
        metric_limit='max_local_scores' if c['method']=='local' else 'max_block_scores'
        if metric_limit not in limits:raise ValueError('scorer requires '+metric_limit)
        unused=({'max_local_scores','max_block_scores','max_fisher_vjps'}&limits.keys())-{metric_limit}-({'max_fisher_vjps'} if c['method']=='fisher_vjp_mc' else set())
        if unused:raise ValueError('unused scorer limits: '+str(sorted(unused)))
        b['max_evaluations']=limits[metric_limit];b['max_tokens']=limits.get('max_tokens',4096)
        if c['method']=='fisher_vjp_mc':
            if 'max_fisher_vjps' not in limits:raise ValueError('Fisher requires max_fisher_vjps')
            b['max_probes']=limits['max_fisher_vjps'];c.setdefault('probes',1);c.setdefault('seed',0)
            _integer(c['probes'],'probes');_integer(c['seed'],'seed',0)
            if c['probes']>b['max_probes']:raise ValueError('Fisher probe budget exceeded')
            if 'fisher_positions' in c:
                positions=_list(c['fisher_positions'],'fisher_positions')
                for position in positions:
                    if not isinstance(position,list) or len(position)!=2 or any(type(i) is not int or i<0 for i in position):raise ValueError('Fisher positions must be [batch,time] nonnegative integer pairs')
                if len({tuple(p) for p in positions})!=len(positions) or len(positions)*c['probes']>b['max_probes']:raise ValueError('duplicate/excessive Fisher positions')

        if c['method']!='local':
            if 'checkpoint' not in c:raise ValueError('blockwise/Fisher requires checkpoint')
            c.setdefault('context','reference_inputs');_choice(c['context'],('reference_inputs','selected_quantized_inputs'),'context')
    if actual=='generate':
        q=source['quantization'];_fields(q,('recipe',),('imatrix_path',),'quantization');_mapping(q['recipe'],'recipe')
        for t in q['recipe'].values():_text(t,'qtype')
        c['native_quantizer']=q
        targets=source['targets'];_fields(targets,('include',),('exclude',),'targets')
        if targets['include']!='eligible':_names(targets['include'],'targets.include')
        if 'exclude' in targets:_names(targets['exclude'],'targets.exclude')
        c['targets']=targets
        c['resolved_candidates']=expand_generators(source['candidate_generators'],b['max_candidates'])
        if any(x['generator']=='awq' for x in c['resolved_candidates']) and 'activation_manifest' not in c:raise ValueError('AWQ requires activation_manifest')
        c['parameters']=source.get('parameters',{})
        _fields(c['parameters'],(),('defaults','overrides','locks'),'parameters')
        if 'defaults' in c['parameters']:c['parameters']['defaults']=parameters(c['parameters']['defaults'])
        if 'imatrix_path' in q:_text(q['imatrix_path'],'imatrix_path')
    if actual in ('select','assemble'):
        selection=source['selection']
        if actual=='assemble':
            _fields(selection,('assignment',),label='selection');_mapping(selection['assignment'],'assignment');c['assignment']=selection['assignment']
        else:
            _fields(selection,('policy','baseline','groups'),('locks','max_passes','improvement_tolerance','scores','stages','alternatives'),'selection')
            c['policy']=_choice(selection['policy'],('manual','per_group_score','groupwise_kl','pipeline'),'selection policy')
            _mapping(selection['baseline'],'baseline');c['baseline']=selection['baseline'];c['groups']=selection['groups']
            c['locks']=selection.get('locks',{});c['max_passes']=_integer(selection.get('max_passes',1),'max_passes')
            c['tolerance']=_number(selection.get('improvement_tolerance',0),'improvement_tolerance',high=1e300)
            c['selection_split']='selection'
            if 'scores' in selection:c['scores']=_text(selection['scores'],'scores')
            if 'stages' in selection:
                c['stages']=selection['stages']
                stages=_list(c['stages'],'selection.stages')
                if c['policy']!='pipeline':raise ValueError('stages require pipeline policy')
                for entry in stages[:-1]:
                    _fields(entry,('type','scores','keep','context'),label='filter stage')
                    if entry['type']!='filter' or entry['context']!='frozen':raise ValueError('only explicit frozen-context score filtering supported')
                    _text(entry['scores'],'scores');_integer(entry['keep'],'keep')
                _fields(stages[-1],('type','method'),label='final stage')
                if stages[-1]!={'type':'select','method':'groupwise_kl'}:raise ValueError('pipeline must finish with groupwise_kl')
                if len(stages)<2:raise ValueError('pipeline requires a filter and final selector')
            elif c['policy']=='pipeline':raise ValueError('pipeline requires explicit stages')
            if 'alternatives' in selection:c['alternatives']=selection['alternatives']
            seen=set()
            if not isinstance(c['groups'],dict) or not c['groups']:raise ValueError('groups must be nonempty mapping')
            for name,members in c['groups'].items():
                _text(name,'group');_names(members,'members')
                if seen.intersection(members) or not set(members)<=c['baseline'].keys():raise ValueError('overlapping/unmatched group')
                seen.update(members)
                if set(members)&c['locks'].keys() and not set(members)<=c['locks'].keys():raise ValueError('split groups containing locked and unlocked tensors')
            if seen!=c['baseline'].keys():raise ValueError('groups must cover baseline')
            if not set(c['locks'])<=c['baseline'].keys():raise ValueError('unknown locked tensor')
            if any(c['baseline'][n]!=v for n,v in c['locks'].items()):raise ValueError('baseline differs from locked assignment')
            b['max_evaluations']=limits.get('max_model_evaluations',1);b['max_tokens']=limits.get('max_tokens',4096)
            if c['policy'] in ('groupwise_kl','pipeline') and not {'evaluator','selection_manifest','corpus_partitions_manifest'}<=c.keys():raise ValueError('KLD selection requires evaluator and selection/corpus locks')
            if c['policy'] in ('groupwise_kl','pipeline') and 'max_model_evaluations' not in limits:raise ValueError('KLD selection requires explicit max_model_evaluations')
            if c['policy']=='per_group_score' and 'scores' not in c:raise ValueError('per_group_score requires scores')
            if c['policy'] in ('manual','per_group_score') and 'native_loader' not in c:raise ValueError('assignment selection requires native_loader')
    if actual=='score':
        if c['method']=='local' and any(k in c for k in ('context','checkpoint','probes','seed','fisher_positions')):raise ValueError('local scorer cannot accept blockwise/Fisher controls')
        if c['method']!='fisher_vjp_mc' and any(k in c for k in ('probes','seed','fisher_positions')):raise ValueError('Fisher controls require fisher_vjp_mc')
        if c['selection_policy']=='held_out' and not {'validation_manifest','pairing_manifest'}<=c.keys():raise ValueError('held_out requires validation/pairing manifests')
        if c['method']!='local' and (c['selection_policy']!='in_sample' or any(k in c for k in ('validation_manifest','pairing_manifest'))):raise ValueError('advanced scorer requires its frozen capture context, not local held-out controls')
        if 'assignments' in c:
            if not isinstance(c['assignments'],dict) or not c['assignments']:raise ValueError('assignments must be nonempty mapping')
            for name,assignment in c['assignments'].items():_text(name,'assignment name');_mapping(assignment,'assignment')
            work=sum(len(a) for a in c['assignments'].values()) if c['method']=='local' else len(c['assignments'])
            if work>b['max_evaluations']:raise ValueError('joint observation budget exceeded')
        c.update(validate_execution_config(c,local=c['method']=='local'))
    output=source.get('output',{})
    required=() if actual=='validate' else (('gguf',) if actual=='assemble' else ('directory',))
    _fields(output,required,(),label='output')
    if 'directory' in output:c['output_directory']=_text(output['directory'],'output.directory')
    if 'gguf' in output:c['output_gguf']=_text(output['gguf'],'output.gguf')
    c['budgets']=b;c['resource_bounds']=dict(b,candidates_per_tensor=len(c.get('resolved_candidates',[])))
    return c


def native_kl_execution(value):
    """Native KL numerical policy; CPU remains the default, CUDA is explicit/full-only."""
    _fields(value,(),('backend','cuda_device','gpu_layers','cache_type_k','cache_type_v',
        'flash_attention','cublas_compute_type','mmq_precision'),'native KL execution')
    p=copy.deepcopy(value)
    _choice(p.setdefault('backend','cpu'),('cpu','cuda'),'backend')
    for name in ('cache_type_k','cache_type_v'):
        _choice(p.setdefault(name,'f16'),('f32','f16','bf16'),'native '+name)
    _choice(p.setdefault('flash_attention','auto'),('on','off','auto'),'flash_attention')
    _choice(p.setdefault('cublas_compute_type','auto'),('auto','f32','f16','bf16'),'cublas_compute_type')
    _choice(p.setdefault('mmq_precision','auto'),('auto','q4','q8'),'mmq_precision')
    if p['backend']=='cuda':
        if 'cuda_device' not in p:raise ValueError('CUDA requires an explicit cuda_device index')
        _integer(p['cuda_device'],'cuda_device',0)
        if p.setdefault('gpu_layers','all')!='all':raise ValueError('CUDA KL requires full model offload')
    elif 'cuda_device' in p or 'gpu_layers' in p or p['cublas_compute_type']!='auto' or p['mmq_precision']!='auto':
        raise ValueError('CPU execution cannot request CUDA placement/compute controls')
    return p


def validate_pairing(pairing,train_manifest,val_manifest):
    """Pair independently verified capture partitions; final test cannot enter either."""
    _fields(pairing,('schema_version','source_sha256','split_method','train','validation'),label='pairing')
    if pairing['schema_version']!=2:raise ValueError('unsupported pairing version')
    if pairing['split_method'] not in ('split_before_reservoir','tiny_reservoir'):raise ValueError('unsupported split method')
    for m in (train_manifest,val_manifest):
        if m.get('source_sha256')!=pairing['source_sha256']:raise ValueError('activation source mismatch')
    for role,m in [('train',train_manifest),('validation',val_manifest)]:
        s=pairing[role];_fields(s,('split_id','token_sha256','chunks','row_ids'),label=role)
        _names(s['row_ids'],'row_ids')
        if s['token_sha256']!=m.get('sampling',{}).get('token_sha256'):raise ValueError('split token mismatch')
        for ch in _list(s['chunks'],'chunks'):
            _fields(ch,('token_source_sha256','start','end'),label='chunk')
            _integer(ch['start'],'start',0);_integer(ch['end'],'end')
            if ch['start']>=ch['end']:raise ValueError('empty/reversed chunk')
            if ch['token_source_sha256']!=m.get('sampling',{}).get('corpus_sha256'):raise ValueError('chunk corpus mismatch')
    a,b=pairing['train'],pairing['validation']
    if a['split_id']==b['split_id'] or set(a['row_ids'])&set(b['row_ids']):raise ValueError('split overlap')
    all_chunks=a['chunks']+b['chunks']
    for i,x in enumerate(all_chunks):
        for y in all_chunks[:i]:
            if x['token_source_sha256']==y['token_source_sha256'] and x['start']<y['end'] and y['start']<x['end']:raise ValueError('token-source chunk overlap')
    return copy.deepcopy(pairing)


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

