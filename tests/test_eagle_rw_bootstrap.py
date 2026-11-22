"""Bootstrap witnesses are explicitly synthetic CPU/native-method evidence."""
from dataclasses import replace, FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from experiments.eagle_rw_collector import (
    CollectionPurpose, EagleRWCollectionPlan, EagleRWCalibrationAction, EagleRWCollector,
    EagleRWSourceProvenance, EagleRWCohortDescriptor, validate_export,
)
from src.server.decision_state import candidate_state_for_action, FixedCalibrationActionV1
from flexsds.core.eagle_consumer_profile import decision_state_key, PublicationPlan, digest, canonical
from scalar_eagle_fixtures import snapshot, planner, plan
from scalar_eagle_native_fixture import native_runtime, scheduler_module
from test_eagle_rw_collector import CpuExecutor, tick, rewrite


def calibration(scheduler,k=2):
    p=scheduler._planner
    p.cost_governor.profile=None
    p.cost_governor.profile_error=None
    state=snapshot(p,('r',),sequence=scheduler._schedule_sequence+1,old={'r':k})
    candidate=candidate_state_for_action(p.runtime_identity,state,(('r',k),),k)
    frozen=EagleRWCollectionPlan(EagleRWSourceProvenance.cpu_fixture('c'*40),p.runtime_identity,
        EagleRWCohortDescriptor('calibration-cpu-cohort',tuple(p._request_incarnations.items())),
        decision_state_key(candidate),PublicationPlan(retirements_per_cohort=2,seed=731),
        purpose=CollectionPurpose.CALIBRATION,calibration_action_k=k)
    core=NS(scheduler=scheduler,model_executor=CpuExecutor())
    collector=EagleRWCollector(core,frozen,clock=iter(range(1_000_000,200_000_000,1_000_000)).__next__)
    action=EagleRWCalibrationAction(core,frozen).install()
    # Untimed transition into the predeclared old/future-width cell. No profile
    # is present during either transition or observed retirement.
    for _ in range(2): tick(collector,core.model_executor,[10])
    return collector,core.model_executor,action


@pytest.fixture
def complete_calibration(native_runtime):
    s,_,_,_=native_runtime
    collector,executor,action=calibration(s)
    collector.install()
    for _ in range(3): tick(collector,executor,[10,11,12])
    yield collector,executor
    collector.close(); action.close()


@pytest.mark.parametrize('k',[0,2,3,4])
def test_B1_B2_initial_calibration_without_any_profile(native_runtime,k):
    s,_,_,_=native_runtime
    collector,executor,action=calibration(s,k)
    assert s._planner.cost_governor.profile is None
    collector.install()
    for _ in range(3): tick(collector,executor,[10])
    data=collector.export()
    assert validate_export(data)
    assert [r['canonicality_status'] for r in data['records']]==['NONCANONICAL','CANONICAL','CANONICAL']
    row=data['records'][-1]
    assert row['collection_purpose']=='CALIBRATION'
    assert row['calibration_action_k']==row['planned_worker_k']==row['physical_draft_width_when_observed']==k
    assert row['selected_cost_cap'] is None
    assert row['actual_useful_emitted_tokens']=={'r':1}
    assert data['consumer_profile'] is False and data['canonical_profile_input'] is True
    collector.close(); action.close()


def test_B2_online_strict_missing_profile_still_degrades():
    p,s=planner(ready=False); p.cost_governor.profile=None
    result=plan(p,s)
    assert result.worker_k==0 and result.cost_decision.decision_reason=='missing_profile'
    assert not result.cost_decision.cost_policy_applied


def test_B3_calibration_does_not_call_online_scorer(monkeypatch):
    p,s=planner(ready=False); p.cost_governor.profile=None
    def forbidden(**kwargs): raise AssertionError('No calibration scoring permitted')
    monkeypatch.setattr(p.cost_governor,'decide',forbidden)
    result=p.plan_step(('A','B'),2,0.,decision_state=s,calibration_action=FixedCalibrationActionV1(2,'a'*64))
    assert result.worker_k==2 and result.per_request_k=={'A':2,'B':2}
    assert result.calibration_action.k==2 and result.cost_decision.selected_cap is None


@pytest.mark.parametrize('cap',[0,2,3,4])
def test_B4_identical_host_state_action_has_identical_policy_key(cap):
    p,s=planner()
    state=dict(p._candidate_decision_states(s,{'A':4,'B':2},()))[cap]
    explicit=candidate_state_for_action(p.runtime_identity,s,state.candidate_future_width_by_request,state.candidate_shared_worker_k)
    assert decision_state_key(explicit)==decision_state_key(state)


def test_B5_no_policy_claim_or_survival_contamination(complete_calibration):
    c,_=complete_calibration
    row=c.records[-1].to_data(); d=row['decision_binding']['decision']
    assert all(d[k] is False for k in ('cost_policy_applied','cost_optimal','hardware_cost_claim_allowed'))
    assert d['candidates']==[] and d['calibration_plan_hash']==c.plan.collection_plan_hash
    assert row['feedback_finalizations']['r']['observation']['source_decision_mode']=='calibration'
    finals=[json.loads(line) for line in Path(c.planner.step_telemetry_path).read_text().splitlines()]
    calibration_finals=[r for r in finals if r['record_type']=='cost_verification_finalized' and r['source_decision_mode']=='calibration']
    assert calibration_finals and all(r['normal_survival_counted'] is False for r in calibration_finals)


def test_B6_policy_validation_missing_profile_still_noncanonical(native_runtime):
    s,_,_,_=native_runtime
    c,executor,action=calibration(s)
    action.close()
    policy=replace(c.plan,purpose=CollectionPurpose.POLICY_VALIDATION,calibration_action_k=None)
    c=EagleRWCollector(c.core,policy,clock=iter(range(1_000_000,20_000_000,1_000_000)).__next__).install()
    for _ in range(3): tick(c,executor,[10])
    assert not any(r.canonicality_status.value=='CANONICAL' for r in c.records)
    assert all('unsupported_decision_state' in r.reasons for r in c.records)
    c.close()


@pytest.mark.parametrize('mutation',['purpose','cost_policy_applied','cost_optimal','hardware_cost_claim_allowed','calibration_action_k','calibration_plan_hash','source_git_sha'])
def test_B7_manually_changed_purpose_or_flag_cannot_admit(complete_calibration,mutation):
    c,_=complete_calibration
    def mutate(row):
        if mutation=='purpose': row['collection_purpose']='POLICY_VALIDATION'
        elif mutation in ('cost_policy_applied','cost_optimal','hardware_cost_claim_allowed'):
            row['decision_binding']['decision'][mutation]=True
        elif mutation=='source_git_sha': row.pop(mutation)
        else: row[mutation]=4 if mutation=='calibration_action_k' else '0'*64
    with pytest.raises(ValueError): validate_export(rewrite(c.export(),mutate))


def test_B8_retirements_are_not_independent_cohorts(complete_calibration):
    c,_=complete_calibration
    assert len(c.records)==3
    assert {r.to_data()['cohort_id'] for r in c.records}=={'calibration-cpu-cohort'}
    assert len({r.to_data()['retirement_window_id'] for r in c.records})==3
    second=replace(c.plan,cohort=EagleRWCohortDescriptor('fake-next-cohort',c.plan.cohort.request_incarnations))
    c.close()
    with pytest.raises(ValueError,match='cohort_not_independent'):
        EagleRWCollector(c.core,second).install()


def test_frozen_action_and_wrong_purpose_are_rejected(complete_calibration):
    c,_=complete_calibration
    with pytest.raises(FrozenInstanceError): c.plan.calibration_action_k=4
    with pytest.raises(ValueError): replace(c.plan,calibration_action_k=4)
    with pytest.raises(ValueError): replace(c.plan,purpose=CollectionPurpose.POLICY_VALIDATION)
    with pytest.raises(ValueError): replace(c.plan,purpose='CALIBRATION')


def test_safety_reduction_cannot_be_charged_to_requested_cell(native_runtime,monkeypatch):
    s,_,_,_=native_runtime
    c,executor,action=calibration(s)
    controller=s._planner.adapter.get_controller('r')
    original=controller.select_budget
    def safe(*args,**kwargs):
        d=original(*args,**kwargs); d.k=0; d.num_nodes=0; d.memory_state='pressure'; return d
    monkeypatch.setattr(controller,'select_budget',safe)
    c.install()
    for _ in range(3): tick(c,executor,[10])
    assert not any(r.canonicality_status.value=='CANONICAL' for r in c.records)
    row=c.records[-1].to_data()
    assert row['calibration_action_k']==2 and row['K_applied']=={'r':0} and row['physical_draft_width_when_observed']==0
    c.close(); action.close()


def trajectory(module,patch,path,enabled,terminal):
    path.mkdir()
    runtime=native_runtime.__wrapped__(module,patch,path)
    s,request,_,_=runtime
    c,executor,action=calibration(s)
    if enabled: c.install()
    rows=[]
    for i in range(4):
        if i==3 and terminal=='length': request.max_tokens=request.num_output_tokens+1
        tokens=[10,2,12] if i==3 and terminal=='eos' else [10,11,12]
        schedule,_,result=tick(c,executor,tokens)
        p=s._plan
        rows.append(dict(sequence=s._schedule_sequence,plan=p.plan_sequence,base=p.per_request_base_k,
            applied=p.per_request_k,worker=p.worker_k,action=p.calibration_action.k,
            outputs=[(r.request_id,r.new_token_ids,r.finish_reason) for b in result.values() for r in b.outputs]))
    records=[json.loads(line) for line in Path(s._planner.step_telemetry_path).read_text().splitlines()]
    finals=[{k:v for k,v in r.items() if k!='recorded_at'} for r in records if r['record_type']=='cost_verification_finalized']
    data=c.export(); c.close(); action.close()
    return dict(rows=rows,finals=finals,pending=dict(s._planner.pending_feedback.pending)),data


@pytest.mark.parametrize('terminal',['eos','length'])
def test_calibration_action_is_identical_with_observer_off_on(scheduler_module,monkeypatch,tmp_path,terminal):
    with monkeypatch.context() as patch: off,_=trajectory(scheduler_module,patch,tmp_path/'off',False,terminal)
    with monkeypatch.context() as patch: on,data=trajectory(scheduler_module,patch,tmp_path/'on',True,terminal)
    assert off==on
    assert validate_export(data)
    assert data['records'][-1]['finalization_status']=={'r':'TERMINAL_CENSORED'}
    assert data['records'][-1]['actual_useful_emitted_tokens']['r']<3


@pytest.mark.parametrize("field",["physical_draft_width_when_observed","verified_if_available",
    "monotonic_start","collection_plan_hash","feedback_verifier_origin","actual_useful_emitted_tokens"])
def test_calibration_cannot_drop_mandatory_provenance(complete_calibration,field):
    c,_=complete_calibration
    with pytest.raises(ValueError):
        validate_export(rewrite(c.export(),lambda row:row.pop(field)))


def test_purpose_cannot_relabel_policy_record_as_calibration(complete_calibration):
    c,_=complete_calibration
    data=c.export()
    data["collection_plan"]["collection_purpose"]="POLICY_VALIDATION"
    data["collection_plan"]["calibration_action_k"]=None
    data["collection_plan_hash"]=digest(data["collection_plan"])
    for row in data["records"]:
        row["collection_purpose"]="POLICY_VALIDATION"
        row["calibration_action_k"]=None
        row["collection_plan_hash"]=data["collection_plan_hash"]
        row["raw_sha256"]=digest({k:v for k,v in row.items() if k not in ("canonicality_status","unsupported_reasons","raw_sha256")})
    data["raw_sha256"]=digest({k:v for k,v in data.items() if k!="raw_sha256"})
    with pytest.raises(ValueError): validate_export(data)
