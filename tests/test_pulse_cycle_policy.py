"""T01..T32: migrated production path, synthetic CPU costs/execution only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from scalar_eagle_fixtures import identity, snapshot, event
from scalar_eagle_native_fixture import native_runtime, scheduler_module, retire
from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.cost_policy import SurvivalEstimate, PendingVerifierEvent, SurvivalEvent, VerifierEventKey
from src.proposer.pulse_policy import PulseCostGovernor, PulseSurvivalHistory, PulseCycleLease
from src.server.budget_planner import AdaptiveBudgetPlanner
from src.server.decision_state import PulseContextV1, capture_pulse_context
from src.server.lineage import DraftOrigin, Execution
from flexsds.core.eagle_consumer_profile import (
    EaglePulseProfile, PulseCostCell, ConsumerProfileError, canonical, digest, decision_state_key,
    publish_pulse_cost,
)

FIXTURE = json.loads((Path(__file__).parent/'fixtures/secp_pulse_causal.json').read_text())
DOMAIN = canonical(dict(preparation="deterministic-old0-v1",workload="fixed-tokenized-v1",
    warm_runtime_cache="same-engine-warm-v1",sampling="greedy_no_stops_v1",
    queue=dict(capacity=2,driver="native_enginecore_v030",pending="bounded_native_placeholders",continuation=0),
    controller_recovery="base3-default-v1",source_observer="pinned030-pulse-v1",
    cohort_failure_plan="fixed50-no-replacement-v1",seeds=[731]))


def make(costs=None,ready=True,mode="strict",tmp_path=None):
    p = AdaptiveBudgetPlanner(AdaptiveProposerAdapter(controller_kwargs={"window_size":4,
        "initial_alpha":.85,"candidate_budgets":[0,2,3]}),max_k=3,
        replacement_policy=True,cost_policy_mode=mode,
        step_telemetry_path=str(tmp_path/'pulse.jsonl') if tmp_path else None)
    p.bind_runtime_identity(identity());p.bind_request('A',1);p.pulse_domain_json=DOMAIN
    s = snapshot(p,('A',),old={'A':0})
    context = PulseContextV1(DOMAIN,'A',1,1,100,100,0,2,True)
    states = p._candidate_decision_states(s,{'A':3},())
    costs = costs or FIXTURE['surface_A']
    p.cost_governor.profile = EaglePulseProfile('synthetic-pulse',p.runtime_identity,DOMAIN,
        tuple((decision_state_key(state),PulseCostCell.synthetic_fixture(*costs[str(q)])) for q,state in states),True)
    if ready:
        p.pulse_history.bind_domain(digest(json.loads(DOMAIN)))
        # Explicit synthetic completed cycles qualify this test-owned history;
        # no legacy feedback is relabelled or loaded through a production path.
        for seq,a in enumerate(FIXTURE['accepted_full_base3_history'],10):
            observation = PendingVerifierEvent(VerifierEventKey('A',1,seq,seq,seq+1),3,3,3,a,3)
            cycle = dict(status='COMPLETE',purpose='NORMAL',q=3,continuation=0,
                domain_hash=p.pulse_history.domain_hash,request_id='A',incarnation=1,
                retirements=[{},dict(schedule_sequence=seq+1)])
            assert p.pulse_history.add_cycle(cycle,SurvivalEvent(observation,'ELIGIBLE_NONTERMINAL',1+a))
    return p,s,context


def step(p,s,c,kv=.1):
    return p.plan_step(s.eligible_decode_request_ids,len(s.eligible_decode_request_ids),kv,
                       decision_state=s,pulse_context=c)


def continuation(p,s,c):
    q=p.pulse_cycles.lease.q
    s=replace(s,scheduler_sequence=s.scheduler_sequence+1,old_verification_width_by_request=(('A',q),))
    c=replace(c,schedule_sequence=s.scheduler_sequence)
    return step(p,s,c),s,c


def execution(plan):
    s=plan.decision_state
    rid=next(iter(plan.per_request_k))
    return Execution(object(),{rid:DraftOrigin(rid,dict(s.request_incarnations)[rid],plan.plan_sequence,
        s.scheduler_sequence,plan.per_request_k[rid],plan.worker_k,plan.worker_k,
        plan.per_request_base_k[rid],plan.per_request_source_modes[rid])},schedule_sequence=s.scheduler_sequence)


def complete(p,first,second,a=2,terminal=False):
    rid=next(iter(first.per_request_k));q=first.worker_k
    for plan,tokens in ((first,1),(second,1+a if q else 1)):
        seq=plan.decision_state.scheduler_sequence
        p.record_pulse_proposal(seq,1,plan.worker_k)
        if plan is second and q:
            origin=execution(first).producers[rid]
            p.record_verification(rid,q,a,source=origin,attributed=True,verifier_id=seq)
        rows={0:NS(outputs=[NS(request_id=rid,new_token_ids=[10]*tokens,
                               finish_reason='stop' if terminal and plan is second else None)])}
        p.finalize_cost_feedback(seq,rows)
        p.finalize_pulse_cycle(seq,execution(plan),rows)


def test_T01_q0_reward():
    assert SurvivalEstimate('A',1,3,4,4,(.75,.5,.25)).predicted_cycle_reward(0)==2

def test_T02_q2_reward():
    assert SurvivalEstimate('A',1,3,4,4,(.75,.5,.25)).predicted_cycle_reward(2)==3.25

def test_T03_q3_reward():
    assert SurvivalEstimate('A',1,3,4,4,(.75,.5,.25)).predicted_cycle_reward(3)==3.5

def test_T04_candidate_uses_high():
    p,s,c=make({'0':[1,1,1],'2':[1,1,2],'3':[100,100,100]})
    record=step(p,s,c).cost_decision.to_record()
    row=record['candidates'][1]
    assert row['R_lower']==3.25/2 and not row['supported']

def test_T05_k0_uses_low():
    p,s,c=make({'0':[1,.5,2],'2':[1,1,1],'3':[100,100,100]})
    record=step(p,s,c).cost_decision.to_record()
    assert record['K0_R_upper']==4 and record['selected_cap']==0

def test_T06_strict_positive_support():
    p,s,c=make({'0':[1,1,1],'2':[3.25/2]*3,'3':[100]*3})
    assert step(p,s,c).cost_decision.selected_cap==0

def test_T07_tie_smaller_q():
    p,s,c=make({'0':[1]*3,'2':[3.25/3]*3,'3':[3.5/3]*3})
    assert step(p,s,c).batch_k==2

def test_T08_no_supported_positive():
    p,s,c=make({'0':[1]*3,'2':[100]*3,'3':[100]*3})
    assert step(p,s,c).batch_k==0

def witness():
    runs=[];inputs=[];controllers=[]
    for name in ('surface_A','surface_B'):
        p,s,c=make(FIXTURE[name])
        inputs.append(canonical(dict(state=asdict(s),context=asdict(c),identity=asdict(p.runtime_identity),
            survival=asdict(p.pulse_history.estimate('A',1,3)))))
        controllers.append(dict(vars(p.adapter.get_controller('A'))))
        runs.append(step(p,s,c))
    assert inputs[0]==inputs[1]
    assert controllers[0]==controllers[1]
    return runs

def test_T09_surface_A_final_k():
    assert witness()[0].batch_k==FIXTURE['expected_final_K_A']

def test_T10_surface_B_final_k():
    assert witness()[1].batch_k==FIXTURE['expected_final_K_B']

def test_T11_cost_only_change_flips_final_K(tmp_path):
    a,b=witness()
    assert a.batch_k!=b.batch_k
    assert a.per_request_base_k==b.per_request_base_k=={'A':3}
    assert a.candidate_decision_states==b.candidate_decision_states
    (tmp_path/'causal-witness.json').write_text(json.dumps(dict(inputs=FIXTURE,
        surface_A=a.cost_decision.to_record(),surface_B=b.cost_decision.to_record(),
        final_K_A=a.batch_k,final_K_B=b.batch_k),indent=2))

@pytest.mark.parametrize('kind',['component','EAGLE-RW-v1.1-candidate','first-response'])
def test_T12_T13_T14_nonpulse_profiles_rejected(kind,tmp_path):
    p,s,c=make();data=p.cost_governor.profile.to_data();data['objective']=kind;data['protocol']=kind
    with pytest.raises(ConsumerProfileError):EaglePulseProfile.from_data(data)
    path=tmp_path/'profile.json';path.write_text(json.dumps(data))
    loaded=AdaptiveBudgetPlanner.from_env({'ADAPTIVE_REPLACEMENT_POLICY':'1','ADAPTIVE_COST_POLICY_MODE':'strict',
        'ADAPTIVE_CONSUMER_PROFILE_PATH':str(path)})
    assert loaded.cost_governor.profile is None

@pytest.mark.parametrize('old',[2,3])
def test_T15_diagonal_rejected(old):
    p,s,c=make();s=replace(s,old_verification_width_by_request=(('A',old),))
    assert step(p,s,c).batch_k==0

def test_T16_pulse_executes_once():
    p,s,c=make();first=step(p,s,c);second,s,c=continuation(p,s,c)
    assert [first.worker_k,second.worker_k]==[2,0]

def test_T17_next_action_forced_K0():
    p,s,c=make();step(p,s,c);second,_,_=continuation(p,s,c)
    assert second.cost_decision.decision_mode=='cycle_continuation' and second.worker_k==0

def test_T18_adaptive_cannot_override_continuation(monkeypatch):
    p,s,c=make();step(p,s,c)
    monkeypatch.setattr(p.cost_governor,'decide',lambda **kwargs: (_ for _ in ()).throw(AssertionError('rescored')))
    assert continuation(p,s,c)[0].worker_k==0

def test_T19_lease_consumed_once_then_ordinary_control():
    p,s,c=make();first=step(p,s,c);second,s,c=continuation(p,s,c)
    complete(p,first,second)
    assert p.pulse_cycles.lease is None and len(p.pulse_cycles.records)==1
    s=replace(s,scheduler_sequence=3,old_verification_width_by_request=(('A',0),))
    third=step(p,s,replace(c,schedule_sequence=3))
    assert third.worker_k==2 and p.pulse_cycles.active

def test_T20_retirement1_exact_pulse():
    p,s,c=make();first=step(p,s,c);second,_,_=continuation(p,s,c);complete(p,first,second)
    row=p.pulse_cycles.records[-1]
    assert row['status']=='COMPLETE' and row['retirements'][0]['origin']['plan_sequence']==first.plan_sequence

def test_T21_retirement2_exact_continuation():
    p,s,c=make();first=step(p,s,c);second,_,_=continuation(p,s,c);complete(p,first,second)
    row=p.pulse_cycles.records[-1]
    assert row['retirements'][1]['origin']['plan_sequence']==second.plan_sequence
    assert row['retirements'][1]['origin']['request_k']==0

def test_T22_predecessor_cannot_enter_cycle():
    p,s,c=make();step(p,s,c)
    p.finalize_pulse_cycle(0,None,None)
    assert not p.pulse_cycles.lease.record['retirements']

def test_T23_retirement3_cannot_enter_cycle():
    p,s,c=make();first=step(p,s,c);second,s,c=continuation(p,s,c)
    third=step(p,replace(s,scheduler_sequence=3,old_verification_width_by_request=(('A',0),)),
               replace(c,schedule_sequence=3))
    assert third.worker_k==0
    p.finalize_pulse_cycle(3,execution(third),None)
    assert not p.pulse_cycles.lease.record['retirements']
    complete(p,first,second)
    assert len(p.pulse_cycles.records[-1]['retirements'])==2

@pytest.mark.parametrize('defect',['terminal','tail','unknown_sampling','wrong_epoch','wrong_origin'])
def test_T24_terminal_censored_and_support_rules(defect):
    p,s,c=make()
    if defect=='tail':
        assert step(p,s,replace(c,remaining_output=5)).worker_k==0;return
    if defect=='unknown_sampling':
        assert step(p,s,replace(c,greedy_no_stops=False)).worker_k==0;return
    first=step(p,s,c);second,_,_=continuation(p,s,c)
    if defect=='terminal':
        complete(p,first,second,terminal=True)
        assert p.pulse_cycles.records[-1]['status']=='CENSORED'
        assert p.pulse_cycles.records[-1]['raw_poststop_outputs'][-1]['outputs'][0]['finish_reason']=='stop'
    else:
        entry=execution(first)
        entry.producers['A']=replace(entry.producers['A'],**({'incarnation':2} if defect=='wrong_epoch' else {'plan_sequence':999}))
        p.record_pulse_proposal(1,1,first.worker_k)
        p.finalize_pulse_cycle(1,entry,{0:NS(outputs=[NS(request_id='A',new_token_ids=[10],finish_reason=None)])})
        assert p.pulse_cycles.records[-1]['status']=='INVALID'

def test_T25_failure_stays_failed():
    p,s,c=make();first=step(p,s,c);second,_,_=continuation(p,s,c)
    p.finalize_pulse_cycle(1,None,None,invalid_status='timeout')
    complete(p,first,second)
    assert p.pulse_cycles.records[-1]['status']=='INVALID'
    assert publish_pulse_cost([dict(p.pulse_cycles.records[-1],cohort_id='failed-slot')])['complete'] is False
    from experiments.eagle_rw_collector import EaglePulseCycleCollector, EagleRWSourceProvenance, EagleRWCohortDescriptor
    p,s,c=make()
    observer=EaglePulseCycleCollector(NS(scheduler=NS(_planner=p)),EagleRWSourceProvenance.cpu_fixture('c'*40),
        EagleRWCohortDescriptor('failed-calibration-slot',(('A',1),)),2,DOMAIN).install()
    with pytest.raises(ValueError,match='not_applied'):
        step(p,s,c,kv=.95)
    observer.close()
    raw=observer.export()
    assert not raw['canonical_profile_input'] and raw['records'][0]['status']=='INVALID'

def test_T26_calibration_not_normal_survival():
    p,s,c=make(ready=False)
    assert not p.pulse_history.add(event(a=3,d=3,base=3,mode='calibration'))
    assert p.pulse_history.estimate('A',1,3).count==0

def test_T27_A10_not_A10G():
    p,s,c=make();obs=dict(p.runtime_identity.observed);obs['gpu_name']='NVIDIA A10G'
    p._runtime_identity=replace(p.runtime_identity,observed=tuple(obs.items()))
    assert step(p,s,c).cost_decision.decision_reason=='identity_mismatch'

@pytest.mark.parametrize('scope',['telemetry','publication','cohort'])
def test_T28_cycle_telemetry_reconstructs(tmp_path,scope):
    if scope!='telemetry':
        from copy import deepcopy
        from experiments.eagle_rw_collector import EaglePulseCycleCollector, EagleRWSourceProvenance, EagleRWCohortDescriptor
        p,s,c=make(ready=False)
        observer=EaglePulseCycleCollector(NS(scheduler=NS(_planner=p)),EagleRWSourceProvenance.cpu_fixture('c'*40),
            EagleRWCohortDescriptor('cohort-1',(('A',1),)),3,DOMAIN).install()
        first=step(p,s,c);second,_,_=continuation(p,s,c);complete(p,first,second,a=3)
        observer.close()
        assert p.pulse_history.estimate('A',1,3).count==0
        exported=observer.export()
        assert exported['canonical_profile_input'] and not exported['consumer_profile']
        with pytest.raises(ValueError,match='reused'):
            EaglePulseCycleCollector(NS(scheduler=NS(_planner=p)),EagleRWSourceProvenance.cpu_fixture('c'*40),
                EagleRWCohortDescriptor('cohort-2',(('A',1),)),3,DOMAIN).install()
        if scope=='cohort':return
        cycles=[]
        for i in range(50):
            # Synthetic publication-contract test, never live cost evidence.
            row=deepcopy(exported['records'][0]);row.update(cohort_id=f'cpu-{i}',synthetic=False,clock_domain=f'cpu-{i}')
            cycles.append(row)
        report=publish_pulse_cost(cycles)
        assert report['complete'] and report['C_point']==pytest.approx(cycles[0]['cost_ms'])
        assert report['unit']=='ms/completed_two_retirement_cycle' and report['seed']==731
        bad=deepcopy(cycles);bad[0]['pulse_receipt']['decision_state']['old_verification_width_by_request']=[['A',2]]
        with pytest.raises(ConsumerProfileError):publish_pulse_cost(bad)
        return
    p,s,c=make(tmp_path=tmp_path);first=step(p,s,c);second,_,_=continuation(p,s,c);complete(p,first,second)
    rows=[json.loads(line) for line in (tmp_path/'pulse.jsonl').read_text().splitlines()]
    final=next(r for r in rows if r['record_type']=='pulse_cycle_finalized')
    assert final['status']=='COMPLETE' and final['useful_tokens']==4
    assert final['cost_ms']==(final['end_ns']-final['start_ns'])/1e6
    assert final['initial_key']==final['decision']['candidates'][1]['decision_state_key_json']
    assert [r['origin']['request_k'] for r in final['retirements']]==[2,0]

def test_T29_memory_dominates():
    p,s,c=make();assert step(p,s,c,kv=.95).worker_k==0
    p,s,c=make();step(p,s,c)
    q=p.pulse_cycles.lease.q
    s=replace(s,scheduler_sequence=2,old_verification_width_by_request=(('A',q),))
    second=step(p,s,replace(c,schedule_sequence=2),kv=.95)
    assert second.worker_k==0 and not p.pulse_cycles.active

def test_T30_worker_equals_max_applied():
    for costs in (FIXTURE['surface_A'],FIXTURE['surface_B']):
        p,s,c=make(costs);first=step(p,s,c);second,_,_=continuation(p,s,c)
        for result in (first,second):
            assert result.worker_k==max(result.per_request_k.values())
            assert all(k<=result.per_request_base_k[r] for r,k in result.per_request_k.items())
    p,s,c=make()
    p.recovery_hold_steps=3;p._recovery_cap_k=0;p._cohort={('A',1)}
    clipped=step(p,s,c)
    assert clipped.worker_k==0 and not p.pulse_cycles.active
    assert clipped.cost_decision.decision_reason=='pulse_not_applied_exactly'
    assert not clipped.cost_decision.cost_policy_applied and not clipped.cost_decision.hardware_cost_claim_allowed

def test_T31_G1C_transport_and_active_loader(tmp_path):
    p,s,c=make();path=tmp_path/'pulse.json';path.write_text(json.dumps(p.cost_governor.profile.to_data()))
    loaded=AdaptiveBudgetPlanner.from_env({'ADAPTIVE_REPLACEMENT_POLICY':'1','ADAPTIVE_COST_POLICY_MODE':'strict',
        'ADAPTIVE_CONSUMER_PROFILE_PATH':str(path),'ADAPTIVE_PULSE_DOMAIN_JSON':DOMAIN})
    assert isinstance(loaded.cost_governor.profile,EaglePulseProfile)
    request=NS(request_id='A',sampling_params=NS(temperature=0,ignore_eos=True,eos_token_id=None,stop=[],stop_token_ids=[],
        repetition_detection=None),num_output_placeholders=3,max_tokens=100,num_output_tokens=1,
        num_computed_tokens=127,use_structured_output=False)
    assert capture_pulse_context(DOMAIN,s,request,4096,2)==replace(c,remaining_output=99,remaining_model=3969,pending_output_bound=3)
    assert capture_pulse_context(DOMAIN,s,request,4096,None) is None
    request.sampling_params.eos_token_id=2
    assert capture_pulse_context(DOMAIN,s,request,4096,2) is None  # Native stop checks the effective ID.

@pytest.mark.parametrize('q',[None,0,2,3])
def test_T32_G1E_native_feedback_and_warmup(native_runtime,q):
    scheduler,request,_,_=native_runtime
    p=scheduler._planner
    # Replace explicit legacy fixture governor with the active production type.
    p.cost_governor=PulseCostGovernor('strict')
    p.pulse_domain_json=DOMAIN
    p.pulse_history=PulseSurvivalHistory()
    p.pulse_history.bind_request('r',p._request_incarnations['r'],p.adapter.get_controller('r').window_size)
    request.spec_token_ids=[]
    request.sampling_params.temperature=0;request.sampling_params.ignore_eos=True
    request.sampling_params.eos_token_id=None
    request.sampling_params.stop=[];request.sampling_params.stop_token_ids=[]
    scheduler._pulse_queue_capacity=2
    p.adapter.get_controller('r').reset(initial_alpha=.85)
    p.adapter.get_controller('r').candidate_budgets=[0,2,3]
    p.max_k=3
    state=snapshot(p,('r',),old={'r':0},sequence=scheduler._schedule_sequence+1)
    states=p._candidate_decision_states(state,{'r':3},())
    p.cost_governor.profile=EaglePulseProfile('native-synthetic-pulse',p.runtime_identity,DOMAIN,
        tuple((decision_state_key(st),PulseCostCell.synthetic_fixture(*FIXTURE['surface_A'][str(q)])) for q,st in states),True)
    observer=None
    if q is not None:
        from experiments.eagle_rw_collector import EaglePulseCycleCollector, EagleRWSourceProvenance, EagleRWCohortDescriptor
        observer=EaglePulseCycleCollector(NS(scheduler=scheduler),EagleRWSourceProvenance.cpu_fixture('c'*40),
            EagleRWCohortDescriptor('native-pulse',tuple(p._request_incarnations.items())),q,DOMAIN).install()
    expected=3 if q is None else q
    first=scheduler.schedule();firstplan=scheduler._plan
    assert first.num_spec_tokens_to_schedule==expected
    second=scheduler.schedule();secondplan=scheduler._plan
    assert second.num_spec_tokens_to_schedule==0
    p.record_pulse_proposal(firstplan.decision_state.scheduler_sequence,1,expected)
    p.record_pulse_proposal(secondplan.decision_state.scheduler_sequence,1,0)
    retire(scheduler,first,[10]);retire(scheduler,second,list(range(10,11+expected)))
    row=p.pulse_cycles.records[-1]
    assert row['status']=='COMPLETE' and row['useful_tokens']==2+expected
    assert row['normal_survival_counted']==(q is None)
    assert p.pulse_history.estimate('r',p._request_incarnations['r'],3).count==(1 if q is None else 0)
    if observer:
        observer.close();assert observer.export()['canonical_profile_input']
