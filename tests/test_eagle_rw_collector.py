"""CPU/native-method collector integration; every source/worker is synthetic."""
from concurrent.futures import Future
from dataclasses import replace, asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.eagle_rw_collector import (
    EagleRWCollector, EagleRWSourceProvenance, EagleRWCohortDescriptor,
    EagleRWCollectionPlan, EagleRWRetirementRecord, runtime_config_hash,
    validate_export, CollectorCanonicalityStatus,
)
from flexsds.core.eagle_consumer_profile import (
    decision_state_key, PublicationPlan, canonical, digest, EagleConsumerProfile, ConsumerProfileError,
)
from scalar_eagle_fixtures import identity, snapshot, profile_for
from scalar_eagle_native_fixture import native_runtime, retire, scheduler_module


def output_for(schedule,tokens):
    ids=list(schedule.num_scheduled_tokens)
    schedule.num_invalid_spec_tokens=None
    schedule.total_num_scheduled_tokens=sum(schedule.num_scheduled_tokens.values())
    return NS(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},
        sampled_token_ids=[list(tokens) for _ in ids],logprobs=None,prompt_logprobs_dict={},
        pooler_output=None,num_nans_in_logits=None,kv_connector_output=None,ec_connector_output=None,
        cudagraph_stats=None,routed_experts=None,sampling_masks=None)


def settled(runtime,k=2):
    scheduler,request,_,_=runtime
    for _ in range(10): retire(scheduler,scheduler.schedule())
    if k==0:
        p=scheduler._planner
        p.cost_governor.profile=profile_for(p,p._candidate_decision_states(
            snapshot(p,("r",),old={"r":2}),{"r":4},()),(1.,100.,100.,100.))
        # Retain exact additional old-width states reached during transition.
        cells=[]
        for old in (0,2,4):
            state=snapshot(p,("r",),old={"r":old})
            cells.extend(profile_for(p,p._candidate_decision_states(state,{"r":4},()),(1.,100.,100.,100.)).cells)
        p.cost_governor.profile=replace(p.cost_governor.profile,cells=tuple(cells))
    for _ in range(2): retire(scheduler,scheduler.schedule(),[10])
    return scheduler,request


class CpuExecutor:
    def __init__(self):
        self._lineage_pending=None
        self.physical_override=None
        self.calls=[]
        self._lineage_runner=NS(drafter=NS(propose=self.propose))
    def propose(self,rows,width):
        return NS(shape=(rows,width if self.physical_override is None else self.physical_override))
    def execute_model(self,schedule,tokens=(10,11,12)):
        self.calls.append("execute")
        self._lineage_pending=schedule
        self._lineage_runner.drafter.propose(len(schedule.num_scheduled_tokens),schedule.num_spec_tokens_to_schedule)
        return output_for(schedule,tokens)
    def sample_tokens(self):
        return None


def collector_for(scheduler,*,k=2,cohort_id="synthetic-cohort-1",clock=None):
    p=scheduler._planner
    state=snapshot(p,("r",),sequence=scheduler._schedule_sequence+1,
                   old={"r":len(scheduler.requests["r"].spec_token_ids)})
    key=decision_state_key(dict(p._candidate_decision_states(state,{"r":4},()))[k])
    plan=EagleRWCollectionPlan(EagleRWSourceProvenance.cpu_fixture("c"*40),p.runtime_identity,
        EagleRWCohortDescriptor(cohort_id,tuple(p._request_incarnations.items())),key,
        PublicationPlan(retirements_per_cohort=2,seed=731))
    core=NS(scheduler=scheduler,model_executor=CpuExecutor())
    collector=EagleRWCollector(core,plan,clock=clock or iter(range(1_000_000,100_000_000,1_000_000)).__next__)
    return collector,core.model_executor


def tick(collector,executor,tokens=(10,11,12)):
    schedule=collector.scheduler.schedule()
    output=executor.execute_model(schedule,tokens)
    result=collector.scheduler.update_from_output(schedule,output)
    return schedule,output,result


@pytest.fixture
def complete(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    for _ in range(3): tick(collector,executor)
    yield collector,executor
    collector.close()


def test_canonical_complete_retirements_share_declared_independent_cohort(complete):
    collector,_=complete
    rows=collector.export()
    assert validate_export(rows)
    assert rows["consumer_profile"] is False and rows["canonical_profile_input"] is True
    assert [r["canonicality_status"] for r in rows["records"]]==["NONCANONICAL","CANONICAL","CANONICAL"]
    assert {r["cohort_id"] for r in rows["records"]}=={"synthetic-cohort-1"}
    assert len({r["retirement_window_id"] for r in rows["records"]})==3
    row=rows["records"][-1]
    assert row["consumer_cost_ms"]==1. and row["consumer_cost_unit"]=="ms/retired_batch"
    assert row["K_base"]=={"r":4} and row["selected_cost_cap"]==2
    assert row["K_governed"]==row["K_applied"]=={"r":2}
    assert row["planned_worker_k"]==row["physical_draft_width_when_observed"]==2
    assert row["drafted"]==row["accepted"]=={"r":2}
    assert row["verified_if_available"] is None
    assert row["actual_useful_emitted_tokens"]=={"r":3}
    assert row["synthetic"] is True and row["source_git_sha"]=="c"*40
    assert not collector._pending


def rewrite(data,mutate):
    data=json.loads(json.dumps(data))
    row=next(r for r in data["records"] if r["canonicality_status"]=="CANONICAL")
    mutate(row)
    row["raw_sha256"]=digest({k:v for k,v in row.items() if k not in (
        "canonicality_status","unsupported_reasons","raw_sha256")})
    data["raw_sha256"]=digest({k:v for k,v in data.items() if k!="raw_sha256"})
    return data


@pytest.mark.parametrize("field",["decision_state_key","source_git_sha","runtime_config_hash","cohort_id",
    "runtime_identity_snapshot","collection_plan_hash","K_base","selected_cost_cap","K_governed",
    "K_applied","planned_worker_k","old_verification_width","actual_useful_emitted_tokens",
    "feedback_verifier_origin"])
def test_missing_mandatory_binding_is_rejected_even_after_rehash(complete,field):
    collector,_=complete
    bad=rewrite(collector.export(),lambda row:row.pop(field))
    with pytest.raises(ValueError): validate_export(bad)


@pytest.mark.parametrize("mutation",["post_action","request_id_only","configured_quantizer","requested_sku",
    "component_cost","collapsed_k","per_retirement_cohort"])
def test_scientific_negative_controls(complete,mutation):
    collector,_=complete
    def mutate(row):
        if mutation=="post_action": row["decision_binding"]["capture_phase"]="after_worker"
        elif mutation=="request_id_only":
            event=row["feedback_finalizations"]["r"]["observation"]
            event["key"]={"request_id":"r"}
        elif mutation in ("configured_quantizer","requested_sku"):
            obs=dict(row["runtime_identity_snapshot"]["observed"])
            obs["effective_quantization" if mutation=="configured_quantizer" else "gpu_name"]="awq_marlin" if mutation=="configured_quantizer" else "A10G"
            row["runtime_identity_snapshot"]["observed"]=list(obs.items())
        elif mutation=="component_cost": row["timing_boundary"]="target_forward_plus_proposal"
        elif mutation=="collapsed_k": row.update(K_base={"r":2},K_governed={"r":2},K_applied={"r":2},old_verification_width={"r":2})
        else: row["cohort_id"]="fabricated-retirement-cohort"
    with pytest.raises(ValueError): validate_export(rewrite(collector.export(),mutate))


def test_no_manual_flags_or_unissued_records(complete):
    collector,_=complete
    data=collector.export()
    data["consumer_profile"]=True
    with pytest.raises(ValueError): validate_export(data)
    record=collector.records[-1]
    assert EagleRWRetirementRecord(record.payload_json,()).canonicality_status==CollectorCanonicalityStatus.INVALID
    modified=replace(record,payload_json=canonical(dict(json.loads(record.payload_json),source_git_sha=None)))
    assert modified.canonicality_status==CollectorCanonicalityStatus.INVALID
    with pytest.raises(ConsumerProfileError): EagleConsumerProfile.from_data(collector.export())


def test_runtime_hash_semantics_are_order_independent_and_sensitive():
    first=identity()
    reversed_identity=replace(first,observed=tuple(reversed(first.observed)),configured=tuple(reversed(first.configured)))
    assert runtime_config_hash(first)==runtime_config_hash(reversed_identity)
    cfg=dict(first.configured);cfg["max_num_seqs"]+=1
    assert runtime_config_hash(first)!=runtime_config_hash(replace(first,configured=tuple(cfg.items())))


@pytest.mark.parametrize("bad",[None,"","4bb093a","main","x"*40])
def test_source_sha_must_be_immutable_hex(bad):
    with pytest.raises(ValueError): EagleRWSourceProvenance.cpu_fixture(bad)


def test_dirty_unfrozen_and_unverified_source_cannot_be_live_provenance(monkeypatch,tmp_path):
    import experiments.eagle_rw_collector as module
    calls=[]
    def git(argv,**kwargs):
        calls.append(argv)
        return NS(stdout=b"d"*40+b"\n" if "rev-parse" in argv else b" M src/server/decision_state.py\n")
    monkeypatch.setattr(module.subprocess,"run",git)
    with pytest.raises(ValueError,match="dirty_or_unfrozen"): EagleRWSourceProvenance.from_checkout(tmp_path)
    assert not EagleRWSourceProvenance("d"*40,False).valid()


@pytest.mark.parametrize("terminal",["eos","length"])
def test_native_terminal_clipping_is_exact_and_not_a_plus_one(native_runtime,terminal):
    scheduler,request=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    tick(collector,executor)
    if terminal=="length": request.max_tokens=request.num_output_tokens+1
    _,_,result=tick(collector,executor,[10,2,12] if terminal=="eos" else [10,11,12])
    row=collector.export()["records"][-1]
    assert row["canonicality_status"]=="CANONICAL"
    emitted=len(result[0].outputs[0].new_token_ids)
    assert row["actual_useful_emitted_tokens"]=={"r":emitted} and emitted<row["accepted"]["r"]+1
    assert row["finalization_status"]=={"r":"TERMINAL_CENSORED"}
    assert not collector.export()["canonical_profile_input"]
    bad=rewrite(collector.export(),lambda row:row.update(actual_useful_emitted_tokens={"r":3}))
    with pytest.raises(ValueError): validate_export(bad)
    collector.close()


def test_a_zero_positive_d_and_planned_vs_physical_remain_distinct(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    tick(collector,executor,[10]);tick(collector,executor,[10])
    row=collector.records[-1].to_data()
    assert row["canonicality_status"]=="CANONICAL" and row["drafted"]=={"r":2} and row["accepted"]=={"r":0}
    executor.physical_override=4
    tick(collector,executor)
    row=collector.records[-1].to_data()
    assert row["planned_worker_k"]==2 and row["physical_draft_width_when_observed"]==4
    assert row["canonicality_status"]!="CANONICAL"
    collector.close()


def test_explicit_same_engine_k0(native_runtime):
    scheduler,_=settled(native_runtime,k=0)
    collector,executor=collector_for(scheduler,k=0)
    collector.install()
    for _ in range(3): tick(collector,executor,[10])
    data=collector.export();row=data["records"][-1]
    assert row["canonicality_status"]=="CANONICAL"
    assert row["selected_cost_cap"]==row["planned_worker_k"]==row["physical_draft_width_when_observed"]==0
    assert row["K_base"]=={"r":4} and row["old_verification_width"]=={"r":0}
    assert row["drafted"]==row["accepted"]=={"r":0}
    assert row["feedback_verifier_origin"]=={} and row["k0_semantics"]=="same_engine_eagle_zero_width_v1"
    assert validate_export(data)
    collector.close()


@pytest.mark.parametrize("status",["timeout","oom","abort","identity_mismatch","unsupported_decision_state",
    "pairing_failure","observer_failure","invalid_finalization"])
def test_failed_attempts_preserve_available_provenance_and_never_sample(native_runtime,status):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    schedule=scheduler.schedule()
    row=collector.failure(status,schedule,detail="synthetic failure").to_data()
    assert row["failure_status"]==status and row["canonicality_status"]!="CANONICAL"
    assert row["source_git_sha"]=="c"*40 and row["cohort_id"]=="synthetic-cohort-1" and row["plan_sequence"]
    assert row["decision_state_key"] and row["K_base"]=={"r":4}
    assert not collector.export()["canonical_profile_input"]
    collector.close()


def test_observer_clock_error_does_not_change_native_result(native_runtime):
    scheduler,_=settled(native_runtime)
    def broken(): raise RuntimeError("synthetic clock error")
    collector,executor=collector_for(scheduler,clock=broken)
    collector.install()
    _,_,result=tick(collector,executor)
    assert result[0].outputs[0].new_token_ids==[10,11,12]
    row=collector.records[-1].to_data()
    assert row["failure_status"]=="observer_failure"
    assert row["feedback_verifier_origin"]["r"]["origin"]["incarnation"]==1
    assert row["feedback_finalizations"]["r"]["emitted_tokens"]==3
    assert row["physical_draft_width_when_observed"]==2
    collector.close()


def test_naturally_materialized_future_pairs_without_forcing_result(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    original=executor.execute_model
    futures=[]
    def delayed(schedule,tokens=(10,11,12)):
        value=original(schedule,tokens)
        future=Future();futures.append((future,value));return future
    executor.execute_model=delayed
    collector.install()
    schedule=scheduler.schedule()
    future=executor.execute_model(schedule)
    assert not future.done() and not collector.records
    future.set_result(futures[-1][1])
    output=future.result()  # the simulated engine materializes once, not the observer
    scheduler.update_from_output(schedule,output)
    assert collector.records[-1].to_data()["unsupported_reasons"]==["anchor_retirement"]
    collector.close()


def test_canonical_collector_source_has_no_gpu_work_or_synchronization():
    import ast
    import experiments.eagle_rw_collector as module
    tree=ast.parse(Path(module.__file__).read_text())
    prohibited={"synchronize","cpu","numpy","tolist","item","forward","load_model"}
    assert not any(isinstance(n,ast.Call) and getattr(n.func,"attr",None) in prohibited for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and getattr(n.func,"attr",None)=="result" for n in ast.walk(tree))


def scenario(module,patch,path,enabled,terminal,cohort="synthetic-cohort-1"):
    path.mkdir()
    runtime=native_runtime.__wrapped__(module,patch,path)
    scheduler,request=settled(runtime)
    collector,executor=collector_for(scheduler,cohort_id=cohort)
    if enabled: collector.install()
    trajectory=[]
    for index in range(4):
        if index==3 and terminal=="length": request.max_tokens=request.num_output_tokens+1
        tokens=[10,2,12] if index==3 and terminal=="eos" else [10,11,12]
        schedule,_,result=tick(collector,executor,tokens)
        plan=scheduler._plan
        trajectory.append(dict(sequence=scheduler._schedule_sequence,plan=plan.plan_sequence,
            base=plan.per_request_base_k,governed=plan.per_request_governed_k,applied=plan.per_request_k,
            planned_worker=plan.worker_k,selected=plan.cost_decision.selected_cap,
            output=[(r.request_id,list(r.new_token_ids),r.finish_reason) for b in result.values() for r in b.outputs]))
    records=[json.loads(line) for line in Path(scheduler._planner.step_telemetry_path).read_text().splitlines()]
    feedback=[{k:v for k,v in row.items() if k!="recorded_at"}
              for row in records if row["record_type"]=="cost_verification_finalized"]
    exported=collector.export()
    collector.close()
    assert not collector._pending
    return dict(trajectory=trajectory,feedback=feedback,status=request.status.name,
                pending=dict(scheduler._planner.pending_feedback.pending),calls=executor.calls),exported


@pytest.mark.parametrize("terminal",["eos","length"])
def test_canonical_binding_observer_off_on_native_trajectory_equivalence(scheduler_module,monkeypatch,tmp_path,terminal):
    with monkeypatch.context() as patch:
        off,_=scenario(scheduler_module,patch,tmp_path/"off",False,terminal)
    with monkeypatch.context() as patch:
        on,raw=scenario(scheduler_module,patch,tmp_path/"on",True,terminal)
    assert off==on
    assert raw["records"][-1]["finalization_status"]=={"r":"TERMINAL_CENSORED"}
    assert validate_export(raw)


def test_two_declared_cohorts_remain_distinct(scheduler_module,monkeypatch,tmp_path):
    exports=[]
    for index in range(2):
        with monkeypatch.context() as patch:
            _,raw=scenario(scheduler_module,patch,tmp_path/str(index),True,"eos",cohort=f"synthetic-independent-{index}")
            exports.append(raw)
    assert exports[0]["collection_plan_hash"]!=exports[1]["collection_plan_hash"]
    assert {row["cohort_id"] for row in exports[0]["records"]}=={"synthetic-independent-0"}
    assert {row["cohort_id"] for row in exports[1]["records"]}=={"synthetic-independent-1"}


def test_prequeued_plans_cannot_replace_retiring_decision_binding(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    first=scheduler.schedule();first_output=executor.execute_model(first)
    expected=collector._pending[id(first)]["binding"]
    second=scheduler.schedule();second_output=executor.execute_model(second)
    scheduler.update_from_output(first,first_output)
    scheduler.update_from_output(second,second_output)
    row=collector.records[0].to_data()
    assert row["plan_sequence"]==expected.plan_sequence
    assert row["schedule_sequence"]==expected.scheduler_sequence
    assert row["decision_binding"]==json.loads(expected.payload_json)
    assert collector.records[1].to_data()["plan_sequence"]!=row["plan_sequence"]
    collector.close()


def test_future_failure_before_retirement_is_preserved(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    future=Future()
    executor.execute_model=lambda schedule:future
    collector.install()
    schedule=scheduler.schedule();executor.execute_model(schedule)
    future.set_exception(TimeoutError("synthetic timeout"))
    assert collector.records[-1].to_data()["failure_status"]=="timeout"
    assert not collector._pending
    collector.close()


def test_worker_exception_behavior_is_preserved(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    error=TimeoutError("synthetic worker timeout")
    def fail(schedule): raise error
    executor.execute_model=fail
    collector.install()
    schedule=scheduler.schedule()
    with pytest.raises(TimeoutError) as found: executor.execute_model(schedule)
    assert found.value is error
    assert collector.records[-1].to_data()["failure_status"]=="timeout"
    collector.close()


def test_missing_real_execution_pair_does_not_become_canonical(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,executor=collector_for(scheduler)
    collector.install()
    schedule=scheduler.schedule()
    scheduler.update_from_output(schedule,output_for(schedule,[10,11,12]))
    assert "pairing_failure" in collector.records[-1].reasons
    collector.close()


def test_live_provenance_manifest_is_rechecked_outside_timing(monkeypatch,tmp_path):
    import experiments.eagle_rw_collector as module
    for name in module._CODE_PATHS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic source\n")
    def git(argv,**kwargs):
        if "rev-parse" in argv: data=b"d"*40+b"\n"
        elif "status" in argv or "diff" in argv: data=b""
        else: data=b"# synthetic source\n"
        return NS(stdout=data)
    monkeypatch.setattr(module.subprocess,"run",git)
    source=EagleRWSourceProvenance.from_checkout(tmp_path)
    assert source.valid() and not source.synthetic
    (tmp_path/module._CODE_PATHS[0]).write_text("# changed after observation\n")
    assert not source.valid()


def test_assessment_cannot_be_cleared_to_upgrade_anchor(complete):
    collector,_=complete
    anchor=collector.records[0]
    assert replace(anchor,reasons=()).canonicality_status==CollectorCanonicalityStatus.INVALID


def test_missing_incarnation_cannot_define_a_cohort():
    with pytest.raises(ValueError): EagleRWCohortDescriptor("cohort",(("r",None),))
    with pytest.raises(ValueError): EagleRWCohortDescriptor("",(("r",1),))


def test_new_cohort_id_cannot_be_fabricated_for_same_live_request_incarnation(native_runtime):
    scheduler,_=settled(native_runtime)
    first,executor=collector_for(scheduler)
    first.install();tick(first,executor);first.close()
    second,_=collector_for(scheduler,cohort_id="fabricated-next-retirement")
    with pytest.raises(ValueError,match="cohort_not_independent"): second.install()


def test_post_action_capture_is_not_an_approved_binding(native_runtime):
    scheduler,_=settled(native_runtime)
    collector,_=collector_for(scheduler)
    collector.install()
    schedule=scheduler.schedule()
    with pytest.raises(ValueError,match="not_captured_before_action"):
        collector._capture(schedule,scheduler._plan)
    collector.close()


def test_same_cohort_reopened_segment_has_unique_window_ids(native_runtime):
    scheduler,_=settled(native_runtime)
    first,executor=collector_for(scheduler)
    first.install();tick(first,executor);first.close()
    second,executor=collector_for(scheduler)
    second.install();tick(second,executor);second.close()
    assert first.records[0].to_data()["cohort_id"]==second.records[0].to_data()["cohort_id"]
    assert first.records[0].to_data()["retirement_window_id"]!=second.records[0].to_data()["retirement_window_id"]


def test_input_canonicality_flag_cannot_be_forced_from_an_anchor_only(complete):
    collector,_=complete
    data=collector.export()
    data["records"]=data["records"][:1]
    data["canonical_profile_input"]=True
    data["raw_sha256"]=digest({k:v for k,v in data.items() if k!="raw_sha256"})
    with pytest.raises(ValueError,match="manually_forced"): validate_export(data)


def test_invalid_records_cannot_be_dropped_from_profile_input_admission(complete):
    collector,_=complete
    row=collector.records[-1]
    collector._records.append(replace(row,payload_json=canonical(dict(json.loads(row.payload_json),source_git_sha=None))))
    data=collector.export()
    assert data["records"][-1]["canonicality_status"]=="INVALID"
    assert data["canonical_profile_input"] is False
