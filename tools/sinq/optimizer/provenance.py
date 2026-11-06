"""Exact immutable candidate-to-tensor assignments within one permitted run."""
from tools.sinq import export_activations as exporter


def candidate_for(run,store,name,key):
    record=run['tensors'].get(name)
    if record is None:
        raise ValueError('tensor is not permitted by the current candidate run: '+name)
    permitted={record['stock'],*[item['candidate_id'] for item in record['candidates']]}
    if key not in permitted:
        raise ValueError('candidate is not permitted for named tensor in current run: '+name)
    return store.get(key,expected={'source_sha256':run['source_sha256'],'tensor_name':name,
        'recipe_sha256':run['recipe_sha256'],'imatrix_sha256':run['imatrix_sha256']})


def graph_assignment(run,store,assignment,*,layer=None,before_layer=None):
    """Validate all pairs before decoding any weight or loading a reference graph."""
    if not isinstance(assignment,dict):raise ValueError('graph assignment must be a mapping')
    result={}
    for name,key in sorted(assignment.items()):
        match=exporter.PATTERN.fullmatch(name)
        if match is None:raise ValueError('unsupported graph projection: '+name)
        index=int(match.group(1))
        if layer is not None and index!=layer:raise ValueError('joint assignment must replace one captured block')
        if before_layer is not None and index>=before_layer:
            raise ValueError('selected-input baseline must replace upstream blocks only')
        result[name]=candidate_for(run,store,name,key)
    return result


def observations(run,store,config,layers):
    """Named explicit joint assignments, or individual immutable candidate observations."""
    result=[]
    if 'assignments' in config:
        for label,assignment in sorted(config['assignments'].items()):
            if not assignment:raise ValueError('joint assignment cannot be empty')
            first=next(iter(assignment));match=exporter.PATTERN.fullmatch(first)
            if match is None:raise ValueError('unsupported joint projection')
            layer=int(match.group(1))
            if layer not in layers:raise ValueError('joint assignment block has no requested capture')
            candidates=graph_assignment(run,store,assignment,layer=layer)
            result.append((label,layer,candidates))
    else:
        for name,record in sorted(run['tensors'].items()):
            match=exporter.PATTERN.fullmatch(name)
            if match is None:raise ValueError('unsupported graph projection: '+name)
            layer=int(match.group(1))
            if layer not in layers:continue
            for key in [record['stock'],*[c['candidate_id'] for c in record['candidates']]]:
                candidate=candidate_for(run,store,name,key)
                result.append((name+':'+key,layer,{name:candidate}))
    if len(result)>config['budgets']['max_evaluations']:
        raise ValueError('advanced observation work budget exceeded')
    return result
