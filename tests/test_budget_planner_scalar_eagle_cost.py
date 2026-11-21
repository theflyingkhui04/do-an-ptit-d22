from dataclasses import asdict, replace
import json
import pytest
from scalar_eagle_fixtures import planner, plan, profile_for
from flexsds.core.eagle_consumer_profile import EagleConsumerProfile, canonical


def noncost(p,s):
    return canonical(dict(identity=asdict(p.runtime_identity),decision_state=asdict(s),
        alpha={r:p.adapter.get_controller(r).alpha_ema for r in ("A","B")},
        survival={r:asdict(p.survival_history.estimate(r,1,b)) for r,b in (("A",4),("B",2))},
        recovery=(p._recovery_cap_k,p._pending_target_k,p._pending_streak,sorted(p._cohort)),
        safety={r:p.adapter.get_controller(r)._memory_pressure_active for r in ("A","B")}))


def assert_causal(run):
    a,b=run
    assert a.per_request_base_k == b.per_request_base_k == {"A":4,"B":2}
    assert a.candidate_decision_states == b.candidate_decision_states
    assert a.per_request_governed_k == {"A":2,"B":2}
    assert b.per_request_governed_k == {"A":4,"B":2}
    assert a.batch_k == 2 and b.batch_k == 4
    assert a.per_request_k == {"A":2,"B":2} and b.per_request_k == {"A":4,"B":2}


def causal_runs(monkeypatch=None):
    pa,sa=planner(costs=(1.,1.8,2.3,3.2))
    pb,sb=planner(costs=(1.,2.1,2.2,2.25))
    assert noncost(pa,sa) == noncost(pb,sb)
    for rid in ("A","B"):
        assert vars(pa.adapter.get_controller(rid)) == vars(pb.adapter.get_controller(rid))
    assert vars(pa.pending_feedback) == vars(pb.pending_feedback)
    assert vars(pa.survival_history) == vars(pb.survival_history)
    assert pa.adapter.controller_kwargs == pb.adapter.controller_kwargs
    assert (pa.max_k,pa.min_batch_k,pa.recovery_hold_steps,pa.replacement_policy,pa.cost_governor.mode) == (
        pb.max_k,pb.min_batch_k,pb.recovery_hold_steps,pb.replacement_policy,pb.cost_governor.mode)
    a,b=pa.cost_governor.profile.to_data(),pb.cost_governor.profile.to_data()
    for data in (a,b):
        for cell in data["cells"]: cell.pop("cost")
    assert a == b  # Only costs and their derived raw digests differ.
    return (pa,sa),(pb,sb)


def test_f_causal_01_only_consumer_cost_changes_active_planner(monkeypatch):
    calls=[]
    lookup=EagleConsumerProfile.lookup
    def observed(self,state):
        calls.append(state)
        return lookup(self,state)
    monkeypatch.setattr(EagleConsumerProfile,"lookup",observed)
    pairs=causal_runs()
    results=[plan(p,s) for p,s in pairs]
    assert len(calls) == 8
    assert_causal(results)
    for result,expected_costs in zip(results,((1.,1.8,2.3,3.2),(1.,2.1,2.2,2.25))):
        evidence=result.cost_decision.to_record()
        assert [s["predicted_yield_batch"] for s in evidence["candidates"]] == pytest.approx([2.,4.95,5.5,5.85])
        assert [s["C_point"] for s in evidence["candidates"]] == list(expected_costs)
        supported=[s for s in evidence["candidates"] if s["cap"]>0 and s["R_lower"]>evidence["K0_R_upper"]]
        reconstructed=max(supported,key=lambda s:s["R_lower"])["cap"]
        assert reconstructed == evidence["selected_cap"] == result.batch_k
        assert result.batch_k == max(result.per_request_k.values())
        assert all(result.per_request_k[r]<=b for r,b in result.per_request_base_k.items())
        assert not result.cost_decision.hardware_cost_claim_allowed


@pytest.mark.parametrize("mutation",["profile_load_only","discard_lookup_result","ignore_governor_result"])
def test_causal_oracle_rejects_disconnected_cost_paths(monkeypatch,mutation):
    pairs=causal_runs()
    if mutation == "profile_load_only":
        for p,s in pairs: p.cost_governor.mode="off"
    elif mutation == "discard_lookup_result":
        constant=pairs[0][0].cost_governor.profile
        original=EagleConsumerProfile.lookup
        def discard(self,state):
            original(self,state)  # Deliberately read and discard the real surface.
            return original(constant,state)
        monkeypatch.setattr(EagleConsumerProfile,"lookup",discard)
    else:
        for p,s in pairs:
            decide=p.cost_governor.decide
            p.cost_governor.decide=lambda decide=decide,**kw: replace(decide(**kw),selected_cap=None)
    with pytest.raises(AssertionError):
        assert_causal([plan(p,s) for p,s in pairs])


def test_shared_batch_rate_not_sum_of_per_request_rates_and_tie_smaller_cap():
    p,s=planner(costs={0:1.,2:4.95/3,3:5.5/3,4:5.85/3})
    result=plan(p,s)
    assert result.cost_decision.selected_cap == 2
    scores=result.cost_decision.to_record()["candidates"]
    assert all(score["R_lower"] == pytest.approx(3.) for score in scores[1:])
    assert all(score["R_point"] == pytest.approx(sum(y for r,y in score["per_request_yield"])/score["C_point"])
               for score in scores)


def test_legacy_parent_selector_shortcut_is_rejected(monkeypatch):
    from src.proposer.controller import BudgetController
    def trap(*a,**kw):
        raise AssertionError("legacy parent selector used instead of current controller")
    monkeypatch.setattr(BudgetController,"select_budget",trap)
    pairs=causal_runs()
    assert_causal([plan(p,s) for p,s in pairs])
    p,s=planner()
    p.adapter.get_controller("A").select_budget=lambda **kw: BudgetController.select_budget(None,**kw)
    with pytest.raises(AssertionError,match="legacy parent selector"):
        plan(p,s)
