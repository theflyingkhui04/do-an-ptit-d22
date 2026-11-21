"""G4 CPU/static checks; adapter success does not certify a canonical collector."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from experiments.scalar_cost_capture import RetirementObserver
from flexsds.core.eagle_consumer_profile import (
    decision_state_key, Cohort, PublicationPlan, publish_cost,
    EagleConsumerProfile, ConsumerProfileError,
)
from scalar_eagle_native_fixture import native_runtime, retire, scheduler_module


RUNTIME=dict(vllm="0.30.0",executor="uni",runner="V1",tp=1,pp=1,
             scheduler_base="AdaptiveAsyncScheduler",async_scheduling=True)


def trajectory(module,patch,path,observer_on,terminal):
    path.mkdir()
    scheduler,request,_,_=native_runtime.__wrapped__(module,patch,path)
    ticks=iter(range(1_000_000,10_000_000,1_000_000))
    observer=RetirementObserver(RUNTIME,clock=lambda:next(ticks),intervals=2)
    if observer_on:
        delegate=scheduler.update_from_output
        def observed(schedule,output):
            return observer.update(delegate,schedule,output,unsafe_requests=())
        scheduler.update_from_output=observed
    schedules=[]
    outputs=[]
    keys=[]
    for index in range(4):
        schedule=scheduler.schedule()
        plan=scheduler._plan
        keys.append(tuple((cap,decision_state_key(state)) for cap,state in plan.candidate_decision_states))
        schedules.append(dict(sequence=scheduler._schedule_sequence,
            plan=plan.plan_sequence,base=plan.per_request_base_k,applied=plan.per_request_k,
            worker=plan.worker_k,old={r:len(v) for r,v in schedule.scheduled_spec_decode_tokens.items()},
            pending=len(scheduler._planner.pending_feedback.pending)))
        if observer_on:
            observer.submit(schedule,planned_k=plan.worker_k,planned_per_request=plan.per_request_k,
                placeholder_counts={"r":len(request.spec_token_ids)},
                incarnations={"r":scheduler._planner._request_incarnations["r"]})
            # Explicit synthetic CPU worker evidence; not inferred live physical state.
            old=len(schedule.scheduled_spec_decode_tokens.get("r",()))
            observer.worker_snapshot(schedule,dict(request_ids=("r",),contexts=(127+index,),
                queries=(old+1,),valid_drafts=(old,),context_kind="MEASURED",prefill=False,
                proposer_rows=1,proposer_k=schedule.num_spec_tokens_to_schedule))
        tokens=[10,11,12]
        if index==3:
            if terminal=="length": request.max_tokens=request.num_output_tokens+1
            else: tokens=[10,2,12]
        result=retire(scheduler,schedule,tokens)
        outputs.append([(row.request_id,tuple(row.new_token_ids),row.finish_reason)
                        for batch in result.values() for row in batch.outputs])
    rows=[json.loads(line) for line in Path(scheduler._planner.step_telemetry_path).read_text().splitlines()]
    final=[{k:v for k,v in row.items() if k!="recorded_at"}
           for row in rows if row["record_type"]=="cost_verification_finalized"]
    return dict(schedules=schedules,outputs=outputs,keys=keys,feedback=final,
                status=request.status.name,pending=len(scheduler._planner.pending_feedback.pending)),observer.export()


@pytest.mark.parametrize("terminal",["eos","length"])
def test_observer_off_on_preserves_native_policy_retirement_trajectory(scheduler_module,monkeypatch,tmp_path,terminal):
    with monkeypatch.context() as patch:
        off,_=trajectory(scheduler_module,patch,tmp_path/"off",False,terminal)
    with monkeypatch.context() as patch:
        on,raw=trajectory(scheduler_module,patch,tmp_path/"on",True,terminal)
    assert off==on
    assert on["pending"]==0
    assert on["feedback"][-1]["finalization_status"]=="TERMINAL_CENSORED"
    assert on["feedback"][-1]["actual_emitted_tokens"] < 3
    assert [e["verifier_id"] for e in raw["events"]]==["v1","v2","v3","v4"]
    assert [e["end_ns"] for e in raw["events"]]==[1_000_000,2_000_000,3_000_000,4_000_000]
    assert raw["windows"][0]["retirement_ids"]==["v2","v3"]
    assert raw["windows"][0]["ms_per_batch"]==1.
    assert raw["events"][-1]["status"]=="unsupported"
    assert raw["events"][-1]["reason"]=="LIFECYCLE_UNSAFE"


def test_existing_adapter_export_cannot_be_published_as_canonical_eagle_rw():
    raw=RetirementObserver(RUNTIME).export()
    assert raw["consumer_profile"] is False and raw["evidence_layer"]=="L2_ADAPTER_ONLY"
    # Required canonical provenance is not supplied by this adapter's export.
    required={"source_git_sha","profile_protocol","runtime_identity","decision_state_key",
              "cohort_id","raw_sha256","config_sha256"}
    assert required.isdisjoint(raw)
    # Flipping a flag cannot make adapter-only output pass real profile admission.
    for claimed in (False,True):
        with pytest.raises(ConsumerProfileError,match="component_or_unknown_profile_protocol"):
            EagleConsumerProfile.from_data(dict(raw,consumer_profile=claimed))
    # An empty/uncollected adapter result cannot satisfy a 50-cohort publication.
    assert not publish_cost((),PublicationPlan(seed=731))["complete"]


def test_decision_key_stays_pre_action_while_worker_measurement_is_separate():
    from scalar_eagle_fixtures import planner,plan
    from src.server.lineage import DraftOrigin
    p,s=planner()
    result=plan(p,s)
    keys=[decision_state_key(state) for cap,state in result.candidate_decision_states]
    original=asdict(s)
    origin=DraftOrigin("A",1,1,1,2,2,source_base_k=4,source_decision_mode="normal")
    p.record_draft_birth(origin,4)
    assert keys==[decision_state_key(state) for cap,state in result.candidate_decision_states]
    assert asdict(s)==original
    assert result.batch_k==2
    assert dict(result.candidate_decision_states)[4].candidate_shared_worker_k==4


def test_canonical_publisher_does_not_add_missing_retirement_provenance():
    # A structurally accepted Cohort does not itself bind a source/config/key.
    fields=set(Cohort.__dataclass_fields__)
    assert {"cohort_id","clock_domain","start_ns","retirement_end_ns","emitted_tokens"}<=fields
    assert {"source_git_sha","decision_state_key","config_sha256"}.isdisjoint(fields)
    source=(Path(__file__).resolve().parents[1]/"experiments/scalar_live_observer.py").read_text()
    assert 'rid + ":incarnation-1"' in source
    assert "DecisionStateKeyV1" not in source


def test_project_local_preaction_capture_can_join_exact_retiring_feedback(native_runtime):
    """Feasibility witness only: no canonical collector is installed or exported."""
    scheduler,request,_,_=native_runtime
    captured={}
    delegate=scheduler._planner.plan_step
    def capture(*args,**kwargs):
        result=delegate(*args,**kwargs)
        snapshot=result.decision_state
        # Called before _apply_replacement_plan writes this action's future K.
        assert snapshot.scheduler_sequence==scheduler._schedule_sequence
        captured[snapshot.scheduler_sequence]=dict(
            plan=result.plan_sequence,decision=result.cost_decision.to_record(),
            keys={cap:decision_state_key(state) for cap,state in result.candidate_decision_states},
            applied=dict(result.per_request_k),worker=result.worker_k)
        return result
    scheduler._planner.plan_step=capture
    # Reach normal cost scoring via real native nonterminal finalization.
    for _ in range(10):
        retire(scheduler,scheduler.schedule())
    first=scheduler.schedule()
    first_entry=scheduler._lineage.get(first)
    first_binding=captured[first_entry.schedule_sequence]
    assert first_binding["decision"]["cost_policy_applied"]
    cap=first_binding["decision"]["selected_cap"]
    score=next(row for row in first_binding["decision"]["candidates"] if row["cap"]==cap)
    assert first_binding["keys"][cap]==score["decision_state_key_json"]
    assert first_binding["worker"]==first.num_spec_tokens_to_schedule
    origin=first_entry.consumed["r"][0]
    second=scheduler.schedule()  # Later planner state must not replace the first binding.
    assert scheduler._lineage.get(second).schedule_sequence!=first_entry.schedule_sequence
    result=retire(scheduler,first,[10,2,12])
    rows=[json.loads(line) for line in Path(scheduler._planner.step_telemetry_path).read_text().splitlines()]
    final=[row for row in rows if row["record_type"]=="cost_verification_finalized"][-1]
    assert final["key"]["verifier_sequence"]==first_entry.schedule_sequence
    assert final["key"]["source_plan_sequence"]==origin.plan_sequence
    assert final["key"]["source_schedule_sequence"]==origin.schedule_sequence
    assert final["actual_emitted_tokens"]==len(result[0].outputs[0].new_token_ids)<3
    assert captured[first_entry.schedule_sequence]==first_binding
