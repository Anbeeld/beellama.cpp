"""Bounded deterministic coordinate descent over immutable full model assignments."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from .resources import atomic_write, canonical_json, digest
from .kl import validate_split
from .schemas import read_json


@dataclass(frozen=True)
class State:
    assignment: tuple
    provenance_json: bytes

    @classmethod
    def create(cls, assignment, provenance):
        if not assignment or any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in assignment.items()):
            raise ValueError('state requires complete named candidate assignment')
        return cls(tuple(sorted(assignment.items())),canonical_json(provenance))

    @property
    def state_id(self):
        return digest({'assignment':self.assignment,'provenance':json.loads(self.provenance_json)})


@dataclass(frozen=True)
class Group:
    name: str
    tensors: tuple
    alternatives: tuple


@dataclass(frozen=True)
class SelectionResult:
    state: State
    score: float
    evaluations: int
    trace: tuple
    budget_exhausted: bool


def coordinate_descent(initial, groups, evaluate, *, checkpoint=None, max_evaluations=100, max_passes=4, tolerance=0.0, split='selection', locks=None):
    """evaluate(State)->nonnegative full-vocabulary KL. Errors propagate after checkpoint.

    Resume replays deterministic decisions from cached scores, not mutable models.
    Provenance must bind source, corpus/reference/evaluator/device identities.
    """
    validate_split(split)
    if type(max_evaluations) is not int or max_evaluations<1 or type(max_passes) is not int or max_passes<1 or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('invalid coordinate search budget/tolerance')
    base=dict(initial.assignment); seen=set(); names=set(); normalized=[]
    locks=dict(locks or {})
    if not set(locks)<=base.keys() or any(base[n]!=v for n,v in locks.items()):raise ValueError('stale/unmatched locked assignment')
    for g in sorted(groups,key=lambda g:g.name):
        ts=set(g.tensors)
        if not g.name or g.name in names or not ts or len(ts)!=len(g.tensors) or ts & seen or not ts<=set(base):
            raise ValueError('overlapping, duplicate or unmatched group')
        seen.update(ts); names.add(g.name)
        locked=ts&locks.keys()
        if locked and locked!=ts:raise ValueError('split group containing locked and unlocked tensors')
        alts=[{name:base[name] for name in sorted(ts)}]
        for alt in (() if locked else g.alternatives):
            if set(alt)!=ts or any(not isinstance(v,str) or not v for v in alt.values()):
                raise ValueError('group alternatives must assign exactly the group')
            alts.append(dict(sorted(alt.items())))
        normalized.append({'name':g.name,'tensors':sorted(ts),'alternatives':sorted({canonical_json(a):a for a in alts}.values(),key=canonical_json)})
    if seen!=set(base):
        raise ValueError('groups must cover all assigned tensor names')
    signature=digest({'initial':initial.state_id,'groups':normalized,'tolerance':tolerance,'split':split,'max_passes':max_passes,'locks':locks})
    scores={}; events=[]
    if checkpoint and Path(checkpoint).exists():
        if Path(checkpoint).stat().st_size>16 << 20:
            raise ValueError('checkpoint byte budget exceeded')
        saved=read_json(checkpoint)
        if saved.get('schema_version')!=2 or saved.get('signature')!=signature or saved.get('checksum')!=digest({k:v for k,v in saved.items() if k!='checksum'}):
            raise ValueError('checkpoint provenance/checksum mismatch')
        scores=saved['scores']; events=saved['trace']
        if any(not math.isfinite(v) or v<0 for v in scores.values()) or len(scores)>max_evaluations:
            raise ValueError('checkpoint scores/budget invalid')
    best=initial; best_score=float('inf')
    def save():
        if checkpoint:
            data={'schema_version':2,'signature':signature,'scores':scores,'trace':events,'best_assignment':dict(best.assignment),'best_score':None if not math.isfinite(best_score) else best_score}
            data['checksum']=digest(data)
            atomic_write(checkpoint,canonical_json(data),max_bytes=16 << 20,immutable=False)
    def score(state):
        key=state.state_id
        if key in scores: return scores[key]
        if len(scores)>=max_evaluations: return None
        try:
            value=float(evaluate(state))
            if not math.isfinite(value) or value<0: raise ValueError('KL evaluator returned invalid score')
        except Exception as error:
            events.append({'state':key,'status':'failed','error_type':type(error).__name__}); save(); raise
        scores[key]=value
        events.append({'state':key,'assignment':dict(state.assignment),'score':value,'status':'evaluated'})
        save()
        return value
    best_score=score(initial); save()
    exhausted=False
    for _ in range(max_passes):
        changed=False
        for g in normalized:
            current=best; current_score=best_score
            proposals=[]
            for alt in g['alternatives']:
                assignment=dict(current.assignment); assignment.update(alt)
                proposal=State.create(assignment,json.loads(initial.provenance_json))
                if proposal!=current: proposals.append(proposal)
            for proposal in sorted(set(proposals),key=lambda s:s.assignment):
                value=score(proposal)
                if value is None:
                    exhausted=True; break
                # Strict improvement: current state wins ties and tolerance ties.
                if value < best_score-tolerance:
                    best,best_score=proposal,value
            if best!=current:
                changed=True
                event={'status':'accepted','group':g['name'],'state':best.state_id,'score':best_score}
                if event not in events: events.append(event)
                save()
            if exhausted: break
        if exhausted or not changed: break
    save()
    return SelectionResult(best,best_score,len(scores),tuple(events),exhausted)


def filter_groups(initial,groups,records,*,context_identity,keep,locks=None):
    """Rank alternatives under one explicitly frozen scorer/context; no model mutation.

    Baseline is always available to downstream coordinate descent. Locked groups
    have no alternatives. Scores are estimates, not final-model KL guarantees.
    """
    if type(keep) is not int or keep<1:raise ValueError('shortlist count must be positive')
    base=dict(initial.assignment);locks=dict(locks or {});seen=set();result=[]
    if not set(locks)<=base.keys() or any(base[n]!=v for n,v in locks.items()):raise ValueError('stale/unmatched lock')
    metric=None;observed={}
    for r in records:
        required={'assignment','score','context_identity','scorer_configuration_identity','objective','direction','scope'}
        if not required<=r.keys() or r['context_identity']!=context_identity or r['direction']!='minimize' or r['scope'] not in ('tensor','group'):
            raise ValueError('score record context/direction/scope mismatch')
        signature=(r['scorer_configuration_identity'],r['objective'])
        if metric is None:metric=signature
        if signature!=metric:raise ValueError('cannot compare different scorer configurations/objectives')
        score=r['score']
        if type(score) not in (float,int) or not math.isfinite(score) or score<0:raise ValueError('invalid score')
        key=canonical_json(r['assignment'])
        if key in observed:raise ValueError('duplicate score assignment')
        observed[key]=float(score)
    for group in sorted(groups,key=lambda g:g.name):
        members=set(group.tensors)
        if not members or members&seen or not members<=base.keys():raise ValueError('overlapping/unmatched group')
        seen.update(members)
        locked=members&locks.keys()
        if locked and locked!=members:raise ValueError('split group with locked and unlocked members')
        if locked:
            result.append(Group(group.name,group.tensors,()));continue
        current={n:base[n] for n in sorted(members)}
        alternatives={canonical_json(current):current}
        for assignment in group.alternatives:
            if set(assignment)!=members:raise ValueError('alternative must exactly cover group')
            alternatives[canonical_json(assignment)]=dict(assignment)
        if any(k not in observed for k in alternatives):raise ValueError('missing candidate/group score')
        # Stable initial assignment wins numeric ties, then canonical assignment order.
        ranked=sorted(alternatives,key=lambda k:(observed[k],k!=canonical_json(current),k))
        result.append(Group(group.name,group.tensors,tuple(alternatives[k] for k in ranked[:keep])))
    if seen!=base.keys():raise ValueError('groups must cover complete assignment')
    return tuple(result)


def per_group_score_select(initial,groups,records,*,context_identity,locks=None):
    """Choose deterministic independent minima under a frozen surrogate context."""
    shortlist=filter_groups(initial,groups,records,context_identity=context_identity,keep=1,locks=locks)
    assignment=dict(initial.assignment)
    for group in shortlist:
        if group.alternatives:assignment.update(group.alternatives[0])
    return State.create(assignment,json.loads(initial.provenance_json))
