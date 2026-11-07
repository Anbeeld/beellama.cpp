"""Bounded deterministic coordinate descent over immutable full model assignments."""
from dataclasses import dataclass, field
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
    progress: dict = field(default_factory=dict)


# Full assignment evidence for ~3,000 scores across 248 groups exceeds the old
# 16 MiB writer (and 8 MiB reader) limits. Keep all evidence, with a symmetric cap.
_SELECTION_JSON_BYTES = 128 << 20


def coordinate_descent(initial, groups, evaluate, *, checkpoint=None, max_evaluations=100, max_passes=4, tolerance=0.0, split='selection', locks=None, progress_path=None, recipe=None, evaluator_policy_identity=None):
    """evaluate(State)->nonnegative full-vocabulary KL. Errors propagate after checkpoint.

    Resume replays deterministic decisions from cached scores, not mutable models.
    Provenance must bind source, corpus/reference/evaluator/device identities.
    Optional recipe/policy bindings also participate in checkpoint identity.
    Progress counts groups within the current pass; proposal and acceptance
    counts are cumulative across the explicitly recorded passes. Fully locked
    groups are excluded from eligible coverage. Candidate IDs bind group-local
    assignments; state IDs bind the exact full-model scoring context. Current
    assignment exclusions are not counted as exact score-cache deduplication.
    The checkpoint is authoritative if interruption separates its atomic write
    from the optional progress mirror; resume reconstructs progress from scores.
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
    identity={'initial':initial.state_id,'groups':normalized,'tolerance':tolerance,'split':split,'max_passes':max_passes,'locks':locks}
    # Preserve old checkpoint identities when the new optional bindings are absent.
    if recipe is not None: identity['recipe']=recipe
    if evaluator_policy_identity is not None: identity['evaluator_policy_identity']=evaluator_policy_identity
    signature=digest(identity)
    if checkpoint and progress_path and Path(checkpoint).resolve()==Path(progress_path).resolve():
        raise ValueError('checkpoint and progress paths must differ')
    scores={}; events=[]
    if checkpoint and Path(checkpoint).exists():
        saved=read_json(checkpoint,max_bytes=_SELECTION_JSON_BYTES)
        if saved.get('schema_version')!=2 or saved.get('signature')!=signature or saved.get('checksum')!=digest({k:v for k,v in saved.items() if k!='checksum'}):
            raise ValueError('checkpoint provenance/checksum mismatch')
        scores=saved['scores']; events=saved['trace']
        if any(not math.isfinite(v) or v<0 for v in scores.values()) or len(scores)>max_evaluations:
            raise ValueError('checkpoint scores/budget invalid')
    eligible=[g for g in normalized if not set(g['tensors'])<=locks.keys()]
    best=initial; best_score=float('inf')
    progress={'schema_version':1,'recipe':recipe,'pass':1,'eligible_groups':len(eligible),
              'locked_groups':len(normalized)-len(eligible),'groups_attempted':0,'groups_completed':0,
              'current_group':None,'proposals_scored':0,'proposals_deduplicated_exact':0,
              'accepted_changes':0,'evaluations':len(scores),'budget_exhausted':False,
              'selection_model_kld':None,'evaluator_policy_identity':evaluator_policy_identity,
              'status':'IN_PROGRESS','completed_passes':0,'passes':[]}
    active_pass=None
    # Origins make replayed progress distinguish original scores from exact cache
    # reuse without appending duplicate committed score or acceptance records.
    origins={e['state']:(e.get('pass'),e.get('group')) for e in events if e['status']=='evaluated'}
    def save():
        progress['evaluations']=len(scores)
        progress['selection_model_kld']=None if not math.isfinite(best_score) else best_score
        if active_pass is not None:
            progress['groups_attempted']=len(active_pass['per_group'])
            progress['groups_completed']=sum(g['completed'] for g in active_pass['per_group'])
        if checkpoint:
            data={'schema_version':2,'signature':signature,'scores':scores,'trace':events,'best_assignment':dict(best.assignment),'best_score':progress['selection_model_kld'],'group_progress':progress}
            data['checksum']=digest(data)
            atomic_write(checkpoint,canonical_json(data),max_bytes=_SELECTION_JSON_BYTES,immutable=False)
        if progress_path:
            atomic_write(progress_path,canonical_json(progress),max_bytes=_SELECTION_JSON_BYTES,immutable=False)
    def score(state, candidate=None):
        key=state.state_id
        origin=(progress['pass'],progress['current_group']) if candidate is not None else (None,None)
        if key in scores:
            if candidate is not None:
                candidate['status']='scored' if origins.get(key)==origin else 'exact_score_cache'
                candidate['score']=scores[key]
                counter='proposals_scored' if candidate['status']=='scored' else 'proposals_deduplicated_exact'
                progress[counter]+=1
            return scores[key]
        if len(scores)>=max_evaluations: return None
        try:
            value=float(evaluate(state))
            if not math.isfinite(value) or value<0: raise ValueError('KL evaluator returned invalid score')
        except Exception as error:
            if candidate is not None: candidate['status']='failed'
            progress['status']='FAILED'
            events.append({'state':key,'status':'failed','error_type':type(error).__name__}); save(); raise
        scores[key]=value; origins[key]=origin
        event={'state':key,'assignment':dict(state.assignment),'score':value,'status':'evaluated'}
        if candidate is not None:
            event.update({'pass':origin[0],'group':origin[1]})
            candidate.update({'status':'scored','score':value})
            progress['proposals_scored']+=1
        events.append(event)
        return value
    best_score=score(initial); save()
    exhausted=False
    for pass_number in range(1,max_passes+1):
        changed=False
        progress.update({'pass':pass_number,'groups_attempted':0,'groups_completed':0})
        active_pass={'pass':pass_number,'completed':False,'accepted_changes':0,'per_group':[]}
        progress['passes'].append(active_pass)
        for g in eligible:
            current=best
            progress['current_group']=g['name']
            coverage={'group':g['name'],'completed':False,'initial_state_id':current.state_id,'candidates':[]}
            active_pass['per_group'].append(coverage)
            proposals=[]
            for alt in g['alternatives']:
                assignment=dict(current.assignment); assignment.update(alt)
                proposal=State.create(assignment,json.loads(initial.provenance_json))
                candidate={'candidate_id':digest(alt),'assignment':alt,'state_id':proposal.state_id,
                           'status':'current_assignment' if proposal==current else 'pending'}
                if proposal==current: candidate['score']=scores[proposal.state_id]
                coverage['candidates'].append(candidate)
                if proposal!=current: proposals.append((proposal,candidate))
            save()
            for proposal,candidate in sorted(proposals,key=lambda pair:pair[0].assignment):
                value=score(proposal,candidate)
                if value is None:
                    exhausted=True; break
                # Strict improvement: current state wins ties and tolerance ties.
                if value < best_score-tolerance:
                    best,best_score=proposal,value
                save()
            coverage['completed']=not exhausted
            coverage['selected_state_id']=best.state_id
            if best!=current:
                changed=True
                active_pass['accepted_changes']+=1
                progress['accepted_changes']+=1
                event={'status':'accepted','group':g['name'],'state':best.state_id,'score':best_score}
                if event not in events: events.append(event)
            save()
            if exhausted: break
        if exhausted:
            progress.update({'status':'INCOMPLETE_BUDGET','budget_exhausted':True})
            break
        active_pass['completed']=True
        progress['completed_passes']+=1
        progress['current_group']=None
        if not changed:
            progress['status']='COMPLETE_CONVERGED'
            break
        progress['status']='COMPLETE_FIRST_PASS'
        if pass_number<max_passes: progress['status']='IN_PROGRESS'
    save()
    return SelectionResult(best,best_score,len(scores),tuple(events),exhausted,progress)


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
