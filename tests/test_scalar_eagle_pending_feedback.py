from dataclasses import replace
import json
import time
from types import SimpleNamespace as NS
import pytest
from scalar_eagle_fixtures import event, planner, snapshot, plan
from src.proposer.cost_policy import PendingFeedbackStore, FINALIZATION_STATES
from scalar_eagle_native_fixture import native_runtime, retire, CpuRequest, scheduler_module
from test_adaptive_scheduler_glue import add_running


def test_two_pending_origins_no_overwrite_idempotence_future_schedule_not_proof():
    store=PendingFeedbackStore();store.bind_request("A",1)
    a,b=event(seq=1).observation,event(seq=2).observation
    assert store.stage(a) and store.stage(b) and not store.stage(a)
    assert len(store.pending)==2
    assert store.finalize(a.key,"ELIGIBLE_NONTERMINAL",3).emitted_tokens == 3
    assert store.finalize(a.key,"ELIGIBLE_NONTERMINAL") is None
    assert not store.stage(a)
    assert b.key in store.pending
    assert store.finalize(b.key,"TERMINAL_CENSORED",1).finalization_status == "TERMINAL_CENSORED"
    assert not store.pending


@pytest.mark.parametrize("status",sorted(FINALIZATION_STATES))
def test_finalization_statuses_and_closed_epoch(status):
    store=PendingFeedbackStore();store.bind_request("A",1)
    a=event().observation
    assert store.stage(a)
    final=store.finalize(a.key,status)
    assert final.finalization_status==status
    assert store.finalize(a.key,status) is None


def test_binary_nonterminal_finalization_can_be_online_without_emitted_validation():
    p,s=planner(ready=False)
    a=event(seq=30).observation
    assert p.pending_feedback.stage(a)
    finalized=p.pending_feedback.finalize(a.key,"ELIGIBLE_NONTERMINAL")
    assert finalized.emitted_tokens is None
    assert p.survival_history.add(finalized)
    assert p.survival_history.estimate("A",1,4).count==1


def test_request_id_reuse_cannot_consume_or_reset_new_epoch():
    p,s=planner(ready=False)
    a=event(seq=30).observation
    assert p.pending_feedback.stage(a)
    p.finish_request("A")
    assert not p.pending_feedback.pending
    p.bind_request("A",2)
    assert not p.pending_feedback.stage(a)
    assert p.pending_feedback.finalize(a.key,"ELIGIBLE_NONTERMINAL") is None
    assert p.survival_history.estimate("A",2,4).count==0
    p.survival_history.reset_epoch("A",1)
    assert p.survival_history.estimate("A",2,4).count==0


def test_bound_request_cleanup_before_first_plan_releases_controller():
    from src.server.budget_planner import AdaptiveBudgetPlanner
    p=AdaptiveBudgetPlanner(replacement_policy=True,cost_policy_mode="strict")
    p.bind_request("never-planned",1)
    assert p.adapter.active_request_count==1
    p.finish_request("never-planned")
    assert p.adapter.active_request_count==0
    assert not p._request_incarnations and not p.survival_history._active


def test_pending_only_until_exact_post_stop_row_and_ambiguous_fails_closed():
    p,s=planner(ready=False)
    a=event(seq=30).observation
    assert p.pending_feedback.stage(a)
    assert p.survival_history.estimate("A",1,4).count==0
    plan(p,s)
    assert a.key in p.pending_feedback.pending  # Future scheduling is no evidence.
    p.finalize_cost_feedback(31,{0:NS(outputs=[NS(request_id="A",new_token_ids=[1],finish_reason=None)])})
    assert a.key in p.pending_feedback.pending
    row=NS(request_id="A",new_token_ids=[1],finish_reason=None)
    p.finalize_cost_feedback(30,{0:NS(outputs=[row,row])})
    assert not p.pending_feedback.pending and p.survival_history.estimate("A",1,4).count==0


def records(s,kind):
    path=s._planner.step_telemetry_path
    return [r for r in map(json.loads,open(path,encoding="utf-8")) if r["record_type"]==kind]


def test_production_prestop_pending_then_native_nonterminal_commits(native_runtime):
    s,r,base,status=native_runtime
    p=s._planner
    verifier=s.schedule()
    original=p.record_verification
    observed=[]
    def inspect(*args,**kwargs):
        value=original(*args,**kwargs)
        assert len(p.pending_feedback.pending)==1
        assert p.survival_history.estimate("r",p._request_incarnations["r"],4).count==0
        observed.append(next(iter(p.pending_feedback.pending.values())))
        return value
    p.record_verification=inspect
    core=retire(s,verifier)
    assert len(observed)==1 and (observed[0].A,observed[0].D)==(2,4)
    assert core[0].outputs[0].new_token_ids==[10,11,12]
    assert not p.pending_feedback.pending
    assert p.survival_history.estimate("r",p._request_incarnations["r"],4).count==1
    final=records(s,"cost_verification_finalized")[-1]
    assert final["finalization_status"]=="ELIGIBLE_NONTERMINAL"
    assert final["actual_emitted_tokens"]==3 and final["normal_survival_counted"]
    p.finalize_cost_feedback(observed[0].key.verifier_sequence,core)
    assert len(records(s,"cost_verification_finalized"))==1


def test_production_two_prequeued_verifiers_keep_producer_identity(native_runtime):
    s,r,base,status=native_runtime
    first,second=s.schedule(),s.schedule()
    assert not s._planner.pending_feedback.pending
    retire(s,first);retire(s,second)
    finals=records(s,"cost_verification_finalized")
    assert len(finals)==2 and all(f["normal_survival_counted"] for f in finals)
    assert len({f["key"]["source_schedule_sequence"] for f in finals})==2
    assert len({f["key"]["verifier_sequence"] for f in finals})==2


@pytest.mark.parametrize("terminal",["eos","length","stop_token","resumable"])
def test_production_terminal_native_clipping_censored_even_with_future_schedule(native_runtime,terminal):
    s,r,base,status=native_runtime
    verifier=s.schedule();future=s.schedule()
    tokens=[10,2,12] if terminal in ("eos","resumable") else [10,3,12]
    if terminal=="length": r.max_tokens=1;tokens=[10,11,12]
    if terminal=="resumable": r.resumable=True
    core=retire(s,verifier,tokens)
    final=records(s,"cost_verification_finalized")[-1]
    assert final["finalization_status"]=="TERMINAL_CENSORED" and not final["normal_survival_counted"]
    assert final["actual_emitted_tokens"]==len(core[0].outputs[0].new_token_ids)<3
    assert not s._planner.pending_feedback.pending
    if terminal=="resumable":
        assert r.status==status.WAITING_FOR_STREAMING_REQ and r in s.waiting
        assert r.request_id in s.requests


def test_production_preemption_stale_invalid_then_resume_fresh(native_runtime):
    s,r,base,status=native_runtime
    verifier=s.schedule()
    base._preempt_request(s,r,time.monotonic())
    assert r.num_stale_output_tokens>0
    retire(s,verifier)
    finals=records(s,"cost_verification_finalized")
    assert not any(f["normal_survival_counted"] for f in finals)
    assert finals[-1]["finalization_status"]=="PREEMPTED_INVALID"
    r.status=status.RUNNING;s.waiting.remove(r)
    r.num_stale_output_tokens=0;r.num_computed_tokens=r.num_tokens
    s.schedule();fresh=s.schedule();retire(s,fresh)
    assert records(s,"cost_verification_finalized")[-1]["normal_survival_counted"]


def test_production_stale_origin_and_old_cleanup_cannot_affect_reused_id(native_runtime):
    s,r,base,status=native_runtime
    verifier=s.schedule();entry=s._lineage.get(verifier)
    origin,count=entry.consumed["r"]
    entry.consumed["r"]=(replace(origin,incarnation=-1),count)
    retire(s,verifier)
    assert not any(f["normal_survival_counted"] for f in records(s,"cost_verification_finalized"))
    old_epoch=s._planner._request_incarnations["r"]
    r.status=status.FINISHED_ABORTED;s._free_request(r)
    new=CpuRequest("r",status);add_running(s,new);s.schedule()
    new_epoch=s._planner._request_incarnations["r"]
    assert new_epoch!=old_epoch
    s._free_request(r)
    assert s._planner._request_incarnations["r"]==new_epoch
    assert s._planner.survival_history.estimate("r",new_epoch,4).count==0
