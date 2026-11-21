from dataclasses import replace
import json
import pytest
from scalar_eagle_fixtures import planner, plan, snapshot, profile_for, seed
from src.server.budget_planner import AdaptiveBudgetPlanner


def test_default_off_independent_of_adaptive_strict(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_COST_POLICY_MODE",raising=False)
    monkeypatch.setenv("ADAPTIVE_STRICT","1")
    assert AdaptiveBudgetPlanner.from_env().cost_governor.mode == "off"
    monkeypatch.setenv("ADAPTIVE_COST_POLICY_MODE","nonsense")
    with pytest.raises(ValueError):
        AdaptiveBudgetPlanner.from_env()


@pytest.mark.parametrize("mode,expected",[("off",4),("strict",0),("compat",4)])
def test_missing_profile_modes_and_claims(mode,expected):
    p,s=planner(mode=mode)
    p.cost_governor.profile=None
    result=plan(p,s)
    assert result.batch_k == expected
    assert not result.cost_decision.cost_policy_applied
    assert not result.cost_decision.hardware_cost_claim_allowed


@pytest.mark.parametrize("mode",["strict","compat","off"])
def test_memory_safety_dominates_profile_floor_and_mode(mode):
    p,s=planner(mode=mode,floor=4)
    result=plan(p,s,kv=.95)
    assert result.batch_k == 0 and set(result.per_request_k.values()) == {0}
    assert result.cost_decision.decision_mode == "memory_override"
    assert not result.cost_decision.cost_policy_applied


@pytest.mark.parametrize("defect",["mixed","missing_state","missing_k0","identity","missing_positive"])
@pytest.mark.parametrize("mode,expected",[("strict",0),("compat",4)])
def test_degraded_admission_never_claims_cost(defect,mode,expected):
    p,s=planner(mode=mode)
    if defect == "mixed": s=replace(s,pure_decode=False,has_scheduled_prefill=True)
    if defect == "identity":
        p.cost_governor.profile=replace(p.cost_governor.profile,profile_id="different")
        obs=dict(p.runtime_identity.observed);obs["driver_version"]="other"
        p._runtime_identity=replace(p.runtime_identity,observed=tuple(obs.items()))
    if defect in ("missing_k0","missing_positive"):
        cells=p.cost_governor.profile.cells
        p.cost_governor.profile=replace(p.cost_governor.profile,cells=cells[1:] if defect=="missing_k0" else cells[:-1])
    result=plan(p,s) if defect != "missing_state" else p.plan_step(("A","B"),2,.1)
    assert result.batch_k == expected and not result.cost_decision.cost_policy_applied
    assert not result.cost_decision.hardware_cost_claim_allowed


def test_strict_warmup_uses_base_then_activates_not_k0_forever():
    p,s=planner(ready=False,window=2)
    result=plan(p,s)
    assert result.batch_k == 4 and result.cost_decision.decision_mode == "cost_warmup"
    assert not result.cost_decision.cost_policy_applied and not result.cost_decision.cost_optimal
    for rid,b in (("A",4),("B",2)):
        seed(p,rid,b,b,1);seed(p,rid,b,b,2)
        p.adapter.get_controller(rid).reset(initial_alpha=.95 if b==4 else .7)
    result=plan(p,replace(s,scheduler_sequence=2))
    assert result.cost_decision.decision_mode == "normal" and result.cost_decision.cost_policy_applied


def test_valid_probe_is_exploration_and_never_normal_history():
    p,s=planner(base={"A":2},ready=False,window=2)
    c=p.adapter.get_controller("A");c.reset(initial_alpha=.4);c.probe_interval=1
    result=plan(p,s)
    assert result.batch_k == 2 and result.cost_decision.decision_mode == "probe"
    assert result.per_request_source_modes == {"A":"probe"}
    assert not result.cost_decision.cost_policy_applied and not result.cost_decision.cost_optimal
    assert not result.cost_decision.hardware_cost_claim_allowed
    seed(p,"A",2,2,1,mode="probe")
    assert p.survival_history.estimate("A",1,2).count == 0


def test_inconclusive_and_unprofitable_cost_force_zero():
    for costs,reason in (({0:(1.,.5,2.),2:(2.5,1.,4.),3:(3.,1.,5.),4:(3.,1.,5.)},"INCONCLUSIVE_COST"),
                         ((1.,100.,100.,100.),"no_profitable_candidate")):
        p,s=planner(costs=costs)
        result=plan(p,s)
        assert result.batch_k == 0 and result.cost_decision.decision_reason == reason


def test_cost_cap_beats_floor_and_immediate_downward_recovery():
    p,s=planner(base={"A":4},costs=(1.,1.1,2.,4.),floor=4,hold=3)
    # Same ready history, first establish worker4 via the off mode.
    p.cost_governor.mode="off"
    assert plan(p,s).batch_k == 4
    p.cost_governor.mode="strict"
    result=plan(p,replace(s,scheduler_sequence=2))
    assert result.cost_decision.selected_cap == 2 and result.batch_k == 2
    assert result.per_request_base_k["A"] == 4
    states=p._candidate_decision_states(s,{"A":4},())
    p.cost_governor.profile=profile_for(p,states,(1.,2.,2.,1.1))
    assert plan(p,replace(s,scheduler_sequence=3)).batch_k == 2
    assert plan(p,replace(s,scheduler_sequence=4)).batch_k == 2
    assert plan(p,replace(s,scheduler_sequence=5)).batch_k == 4
    p.cost_governor.profile=None
    assert plan(p,replace(s,scheduler_sequence=6)).batch_k == 0


def test_compat_valid_inputs_apply_governed_result():
    p,s=planner(mode="compat")
    result=plan(p,s)
    assert result.batch_k == 2 and result.cost_decision.cost_policy_applied
    assert not result.cost_decision.hardware_cost_claim_allowed  # Synthetic CPU fixture.


@pytest.mark.parametrize("mode",["strict","compat"])
def test_invalid_profile_file_fails_closed_without_startup_crash(tmp_path,mode):
    p,s=planner()
    data=p.cost_governor.profile.to_data()
    data["cells"][0]["state"]=[]
    path=tmp_path/"invalid.json";path.write_text(json.dumps(data))
    loaded=AdaptiveBudgetPlanner.from_env({"ADAPTIVE_REPLACEMENT_POLICY":"1",
        "ADAPTIVE_COST_POLICY_MODE":mode,"ADAPTIVE_CONSUMER_PROFILE_PATH":str(path)})
    assert loaded.cost_governor.profile is None
    assert loaded.cost_governor.profile_error == "invalid_consumer_profile"


def test_empty_cohort_resets_recovery_and_no_stale_request_result():
    p,s=planner(hold=3)
    assert plan(p,s).batch_k==2
    empty=p.plan_step((),0,.1)
    assert empty.batch_k==0 and empty.per_request_k=={}
    assert p._recovery_cap_k is None and not p._cohort


def test_source_root_import_and_profile_load_use_same_consumer_types(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    p,s=planner()
    path=tmp_path/"profile.json";path.write_text(json.dumps(p.cost_governor.profile.to_data()))
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE="1")
    code='''from src.server.budget_planner import AdaptiveBudgetPlanner
import sys
p=AdaptiveBudgetPlanner.from_env({"ADAPTIVE_REPLACEMENT_POLICY":"1",
    "ADAPTIVE_COST_POLICY_MODE":"strict","ADAPTIVE_CONSUMER_PROFILE_PATH":sys.argv[1]})
assert p.cost_governor.profile is not None
assert p.cost_governor.profile.synthetic
'''
    completed=subprocess.run([sys.executable,"-B","-c",code,str(path)],cwd=root,env=env,
        capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
