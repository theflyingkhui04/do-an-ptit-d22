"""CPU contract for the opt-in Hardware-Adaptive replacement policy."""
import itertools
import json

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController
from src.proposer.controller import EmpiricalCostProfile
from src.server.budget_planner import AdaptiveBudgetPlanner
from test_adaptive_scheduler_glue import (
    add_running,
    build,
    make_request,
    retire,
    scheduler_module,
)
from test_hardware_profiler_benchmark import measured_profile


def controller(**kwargs):
    return AdaptiveDraftController(**kwargs)


def planner(**kwargs):
    return AdaptiveBudgetPlanner(replacement_policy=True, recovery_hold_steps=3, **kwargs)


def set_alpha(adapter, rid, alpha):
    state = adapter.get_controller(rid)
    state.reset(initial_alpha=alpha)
    return state


def plan(p, *alphas, kv=0.1):
    ids = [str(i) for i in range(len(alphas))]
    for rid, alpha in zip(ids, alphas):
        set_alpha(p.adapter, rid, alpha)
    return p.plan_step(ids, batch_size=len(ids), kv_usage=kv)


def feedback(c, count=1):
    for _ in range(count):
        c.record_attributed_verification(num_drafted=3, num_accepted=0)


def runtime(scheduler_module, monkeypatch, tmp_path, *, hold=3, **controller_kwargs):
    path = tmp_path / "policy.jsonl"
    scheduler = build(
        scheduler_module, monkeypatch, tmp_path,
        ADAPTIVE_REPLACEMENT_POLICY="1", ADAPTIVE_RECOVERY_HOLD_STEPS=str(hold),
        ADAPTIVE_CONTROLLER_KWARGS=json.dumps(controller_kwargs),
        ADAPTIVE_STEP_TELEMETRY_PATH=str(path),
    )
    return scheduler, path


def events(path, kind):
    return [row for line in path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))["record_type"] == kind]


def test_t_warm_001_zero_width_and_decisions_do_not_count():
    c = controller(min_warmup_samples=3)
    c.update(0, 0)
    c.update(0, 0)
    c.update(0, 3)  # A direct estimator call has no attributed origin.
    for _ in range(3):
        decision = c.select_budget()
        assert decision.verification_sample_count == 0
        assert decision.warmup_active and decision.rule_alpha == 0.8


def test_t_warm_002_zero_acceptance_is_a_real_sample(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, min_warmup_samples=3)
    add_running(s, make_request("a"))
    s.schedule()
    out = s.schedule()
    retire(s, out, [("a", 3, 0)])
    assert s._planner.adapter.get_controller("a").verification_sample_count == 1
    assert events(path, "verification_feedback")[0]["accepted_tokens"] == 0


def test_t_warm_003_duplicate_origin_counts_once(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    s.schedule()
    out = s.schedule()
    other = s.schedule()
    # Adversarial duplicate consumption across two distinct verifier outputs.
    s._lineage.get(other).consumed["a"] = s._lineage.get(out).consumed["a"]
    retire(s, out, [("a", 3, 1)])
    c = s._planner.adapter.get_controller("a")
    ema = c.alpha_ema
    retire(s, other, [("a", 3, 1)])
    assert c.verification_sample_count == 1 and c.alpha_ema == ema
    assert events(path, "verification_feedback_skipped")[-1]["lineage_status"] == "duplicate-origin"


@pytest.mark.parametrize("damage", ["stale", "incarnation", "producer"])
def test_t_warm_004_bad_feedback_does_not_count(scheduler_module, monkeypatch, tmp_path, damage):
    from dataclasses import replace

    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    req = make_request("a")
    add_running(s, req)
    s.schedule()
    out = s.schedule()
    if damage == "stale":
        req.num_stale_output_tokens = 1
    else:
        entry = s._lineage.get(out)
        origin, count = entry.consumed["a"]
        forged = (replace(origin, incarnation=origin.incarnation + 1) if damage == "incarnation"
                  else replace(origin, schedule_sequence=1000))
        entry.consumed["a"] = forged, count
    retire(s, out, [("a", 3, 0)])
    c = s._planner.adapter.get_controller("a")
    # Direct planner callers cannot supply policy evidence without the scheduler.
    s._planner.record_verification("a", 3, 0)
    assert c.verification_sample_count == 0 and c.alpha_ema == 0.8
    assert events(path, "verification_feedback_skipped")


def test_t_warm_005_gate_transitions_at_n():
    c = controller(min_warmup_samples=3)
    feedback(c, 2)
    before = c.select_budget()
    assert (before.verification_sample_count, before.warmup_active, before.rule_alpha, before.k) == (2, True, 0.8, 3)
    feedback(c)
    after = c.select_budget()
    assert after.verification_sample_count == 3 and not after.warmup_active
    assert after.rule_alpha == pytest.approx(0) and after.k == 0


def test_t_warm_006_decision_reports_both_alphas():
    c = controller(min_warmup_samples=3)
    feedback(c)
    decision = c.select_budget()
    assert decision.effective_alpha == pytest.approx(0)
    assert decision.rule_alpha == 0.8 and decision.warmup_active and decision.k == 3


def test_t_warm_007_explicit_reset_restores_initial_policy_state():
    c = controller(min_warmup_samples=1)
    feedback(c)
    assert c.select_budget().k == 0 and c.consecutive_fallback_steps == 1
    c.reset()
    assert c.verification_sample_count == 0 and c.alpha_ema == 0.8
    assert c.consecutive_fallback_steps == 0 and c.get_stats()["reason_counts"] == {}
    assert c.select_budget().warmup_active


def test_t_floor_001_floor_only_applies_to_acceptance_fallback():
    c = controller(initial_alpha=0.5, min_k=2, probe_interval=1)
    decision = c.select_budget()
    assert decision.k == 2 and decision.reason.startswith("Low acceptance floor")
    assert c.consecutive_fallback_steps == 0


@pytest.mark.parametrize("pressure", [{"kv_cache_usage_pct": 95}, {"free_vram_mb": 1400}])
def test_t_floor_002_raw_memory_overrides_floor(pressure):
    decision = controller(initial_alpha=0.5, min_k=2, min_warmup_samples=3).select_budget(**pressure)
    assert decision.k == 0 and decision.memory_state == "pressure" and decision.rule_alpha is None


def test_t_floor_003_cooldown_overrides_floor():
    c = controller(initial_alpha=0.5, min_k=2)
    assert c.select_budget(kv_cache_usage_pct=95).k == 0
    assert [c.select_budget(kv_cache_usage_pct=70).k for _ in range(3)] == [0, 0, 2]


def test_t_floor_004_candidate_normalization_and_runtime_cap():
    c = controller(initial_alpha=0.5, candidate_budgets=[0, 2, 3, 4], min_k=1)
    assert c.select_budget().k == 2
    with pytest.raises(ValueError, match="min_k"):
        controller(candidate_budgets=[0, 2, 3], min_k=4)
    p = planner(max_k=1, adapter=AdaptiveProposerAdapter(controller_kwargs={"min_k": 1}))
    assert plan(p, 0.5).per_request_k == {"0": 0}


def test_t_floor_005_memory_decisions_do_not_advance_probe_cadence():
    c = controller(initial_alpha=0.5, min_k=0, probe_interval=20)
    assert [c.select_budget().k for _ in range(18)] == [0] * 18
    assert c.select_budget(kv_cache_usage_pct=95).k == 0
    assert [c.select_budget(kv_cache_usage_pct=70).k for _ in range(3)] == [0, 0, 0]
    assert c.consecutive_fallback_steps == 19
    assert c.select_budget().reason.startswith("Periodic probe")


@pytest.mark.parametrize(("alpha", "expected"), [(0.50, 0), (0.64, 0), (0.65, 2),
                                                          (0.79, 2), (0.80, 3), (0.89, 3), (0.90, 4)])
def test_t_floor_006_explicit_acceptance_matrix(alpha, expected):
    for warm, floor, memory, probe_due in itertools.product((False, True), (0, 2), (False, True), (False, True)):
        c = controller(initial_alpha=0.8, min_warmup_samples=int(warm), min_k=floor)
        # Pin the estimator at the exact boundary without introducing EMA rounding.
        c.get_effective_alpha = lambda: alpha
        c.consecutive_fallback_steps = 19 if probe_due else 0
        decision = c.select_budget(kv_cache_usage_pct=95 if memory else 10)
        assert decision.k == (0 if memory else 3 if warm else expected or (2 if floor or probe_due else 0))
        assert decision.rule_alpha == (None if memory else 0.8 if warm else alpha)


def test_t_hold_001_increase_waits_for_three_matching_plans():
    p = planner()
    assert plan(p, 0.7).recovery_cap_k == 2
    assert [plan(p, 0.95).recovery_cap_k for _ in range(3)] == [2, 2, 4]


def test_t_hold_002_downward_transition_is_immediate():
    p = planner()
    assert plan(p, 0.95).worker_k == 4
    lower = plan(p, 0.7)
    assert lower.recovery_cap_k == lower.worker_k == 2


def test_t_hold_003_all_zero_ignores_positive_legacy_batch_floor():
    p = planner(min_batch_k=4)
    plan(p, 0.95, 0.95)
    zero = plan(p, 0.5, 0.5)
    assert zero.worker_k == zero.recovery_cap_k == zero.pending_streak == 0
    assert zero.pending_target_k is None


def test_t_hold_004_mixed_rows_do_not_inherit_a_higher_k():
    p = planner()
    plan(p, 0.95, 0.95)
    mixed = plan(p, 0.5, 0.7)
    assert mixed.per_request_selected_k == mixed.per_request_k == {"0": 0, "1": 2}
    assert mixed.worker_k == 2
    with pytest.raises(ValueError, match="per.request"):
        planner(per_request=False)


def test_t_hold_005_exhaustive_two_request_caps_and_memory():
    alpha_by_k = {0: 0.5, 2: 0.7, 3: 0.8, 4: 0.95}
    for a, b, cap, pressure in itertools.product(alpha_by_k, alpha_by_k, alpha_by_k, (False, True)):
        p = planner()
        plan(p, alpha_by_k[cap], alpha_by_k[cap])
        step = plan(p, alpha_by_k[a], alpha_by_k[b], kv=0.95 if pressure else 0.1)
        expected = {"0": min(a, cap), "1": min(b, cap)} if not pressure else {"0": 0, "1": 0}
        assert step.per_request_k == expected and step.worker_k == max(expected.values())


def test_t_hold_006_empty_or_disjoint_cohort_cannot_inherit_cap():
    p = planner()
    assert plan(p, 0.95).worker_k == 4
    p.finish_request("0")
    assert plan(p).worker_k == 0
    for rid, alpha in (("new", 0.5), ("unrelated", 0.7)):
        set_alpha(p.adapter, rid, alpha)
        step = p.plan_step([rid], 1, 0.1)
        assert step.worker_k == {0.5: 0, 0.7: 2}[alpha]


def test_t_hold_007_new_pending_target_resets_streak():
    p = planner()
    plan(p, 0.7)
    steps = [plan(p, a) for a in (0.95, 0.8, 0.95)]
    assert [(s.pending_target_k, s.pending_streak, s.worker_k) for s in steps] == [(4, 1, 2), (3, 1, 2), (4, 1, 2)]


def test_t_hold_008_zero_interrupts_pending_increase():
    p = planner()
    plan(p, 0.7)
    steps = [plan(p, a) for a in (0.95, 0.95, 0.5, 0.95, 0.95, 0.95)]
    assert [s.worker_k for s in steps] == [2, 2, 0, 0, 0, 4]
    assert [s.pending_streak for s in steps] == [1, 2, 0, 1, 2, 0]


def test_t_life_001_finish_drops_request_local_state():
    p = planner()
    plan(p, 0.7)
    c = p.adapter.get_controller("0")
    feedback(c)
    c.select_budget()
    final = p.finish_request("0")
    assert final["verification_sample_count"] == 1 and sum(final["reason_counts"].values()) == 2
    fresh = p.adapter.get_controller("0")
    assert fresh is not c and fresh.alpha_ema == 0.8 and fresh.verification_sample_count == 0
    assert fresh.consecutive_fallback_steps == 0 and fresh.get_stats()["reason_counts"] == {}


def test_t_life_002_empty_and_explicit_reset_clear_recovery():
    p = planner()
    for reset in (lambda: plan(p), p.reset_recovery):
        plan(p, 0.7)
        assert plan(p, 0.95).pending_streak == 1
        reset()
        fresh = plan(p, 0.95)
        assert fresh.recovery_cap_k == 4 and fresh.pending_target_k is None and fresh.pending_streak == 0


def test_t_life_003_preempted_feedback_cannot_advance_warmup(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    a, b = make_request("a"), make_request("b")
    add_running(s, a, b)
    s.schedule()
    out = s.schedule()
    a.num_stale_output_tokens = 3
    retire(s, out, [("a", 3, 3), ("b", 3, 2)])
    assert s._planner.adapter.get_controller("a").verification_sample_count == 0
    assert s._planner.adapter.get_controller("b").verification_sample_count == 1
    assert [e["request_id"] for e in events(path, "verification_feedback")] == ["b"]
    new = s.schedule()
    assert set(s._lineage.get(new).producers) == {"b"} and a.spec_token_ids == []


@pytest.mark.parametrize("finish", [True, False])
def test_t_life_004_request_id_reuse_cannot_inherit_state(scheduler_module, monkeypatch, tmp_path, finish):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, min_warmup_samples=2)
    old = make_request("a")
    add_running(s, old)
    s.schedule()
    retire(s, s.schedule(), [("a", 3, 0)])
    pending = s.schedule()
    old_incarnation = s._lineage.active["a"][1]
    if finish:
        s._free_request(old)
    else:
        s.running.remove(old)
    new = make_request("a")
    add_running(s, new)
    s.schedule()
    if not finish:
        s._free_request(old)
        assert s.requests["a"] is new
    retire(s, pending, [("a", 3, 3)])
    c = s._planner.adapter.get_controller("a")
    assert s._lineage.active["a"][1] != old_incarnation
    assert c.verification_sample_count == 0 and c.alpha_ema == 0.8
    assert c.consecutive_fallback_steps == 0
    assert events(path, "verification_feedback_skipped")


def test_t_act_001_only_actual_eligible_rows_determine_worker(monkeypatch, scheduler_module, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("high"), make_request("mid"), make_request("prefill", prefill=True))
    set_alpha(s._planner.adapter, "high", 0.95)
    set_alpha(s._planner.adapter, "mid", 0.7)
    s.scheduled_ids = {"mid", "prefill"}
    out = s.schedule()
    later = s.schedule()
    assert out.num_spec_tokens_to_schedule == 2
    assert {rid: len(row) for rid, row in later.scheduled_spec_decode_tokens.items()} == {"mid": 2}
    retire(s, later, [("mid", 2, 1)])
    row = events(path, "planned_decision")[0]
    assert row["request_id"] == "mid" and row["selected_k"] == row["applied_k"] == 2
    assert s._test_runner._draft_token_ids.shape == (2, 2)
    assert events(path, "verification_feedback")[0]["physical_draft_width"] == 2
    s.scheduled_ids = {"prefill"}
    assert s.schedule().num_spec_tokens_to_schedule == 0
    s._planner.max_k = 0
    s.scheduled_ids = {"mid"}
    assert s.schedule().num_spec_tokens_to_schedule == 0
    assert events(path, "batch_policy")[-1]["eligible_batch_size"] == 0


def test_t_act_002_mixed_zero_two_trims_only_verified_prefix(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    a, b = make_request("a"), make_request("b")
    add_running(s, a, b)
    set_alpha(s._planner.adapter, "a", 0.5)
    set_alpha(s._planner.adapter, "b", 0.7)
    assert s.schedule().num_spec_tokens_to_schedule == 2
    verifier = s.schedule()
    assert s._test_runner._draft_token_ids.shape == (2, 2)
    birth = events(path, "draft_policy_produced")
    assert [(r["request_id"], r["applied_k"], r["physical_draft_width"]) for r in birth[:2]] == [
        ("a", 0, 2), ("b", 2, 2)]
    assert set(verifier.scheduled_spec_decode_tokens) == {"b"}
    retire(s, verifier, [("b", 2, 0)])
    assert s._planner.adapter.get_controller("a").verification_sample_count == 0
    row = events(path, "verification_feedback")[0]
    assert row["request_id"] == "b" and row["producer_schedule_id"] == 1


def test_t_act_003_downward_worker_change_is_immediate(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    set_alpha(s._planner.adapter, "a", 0.95)
    assert s.schedule().num_spec_tokens_to_schedule == 4
    set_alpha(s._planner.adapter, "a", 0.7)
    assert s.schedule().num_spec_tokens_to_schedule == 2
    assert [r["worker_k"] for r in events(path, "batch_policy")] == [4, 2]


def test_t_act_004_memory_zero_still_verifies_older_origin(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, min_k=2)
    add_running(s, make_request("a"))
    set_alpha(s._planner.adapter, "a", 0.95)
    assert s.schedule().num_spec_tokens_to_schedule == 4
    s.kv_usage = 0.95
    zero = s.schedule()
    assert zero.num_spec_tokens_to_schedule == 0 and s._lineage.generation is None
    retire(s, zero, [("a", 4, 2)])
    origin = events(path, "verification_feedback")[0]
    assert (origin["producer_schedule_id"], origin["worker_k"], origin["applied_k"]) == (1, 4, 4)


def test_t_act_005_recovery_waits_after_memory_cooldown(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    s.schedule()
    s.kv_usage = 0.95
    assert s.schedule().num_spec_tokens_to_schedule == 0
    s.kv_usage = 0.7
    assert [s.schedule().num_spec_tokens_to_schedule for _ in range(5)] == [0, 0, 0, 0, 3]
    batches = events(path, "batch_policy")
    assert [(r["fresh_max_k"], r["worker_k"]) for r in batches[-3:]] == [(3, 0), (3, 0), (3, 3)]


def test_t_act_006_reordered_rows_follow_their_origins(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"), make_request("b"))
    set_alpha(s._planner.adapter, "a", 0.7)
    set_alpha(s._planner.adapter, "b", 0.95)
    s.schedule()
    s.running.reverse()
    retire(s, s.schedule(), [("b", 4, 4), ("a", 2, 0)])
    assert [(r["request_id"], r["producer_schedule_id"], r["applied_k"])
            for r in events(path, "verification_feedback")] == [("b", 1, 4), ("a", 1, 2)]
    assert s._planner.adapter.get_controller("a").get_effective_alpha() == pytest.approx(0)
    assert s._planner.adapter.get_controller("b").get_effective_alpha() == pytest.approx(1)


def test_t_act_007_multiple_inflight_use_consumed_producer_not_latest(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, hold=0)
    add_running(s, make_request("a"))
    set_alpha(s._planner.adapter, "a", 0.7)
    s.schedule()
    set_alpha(s._planner.adapter, "a", 0.95)
    second, third = s.schedule(), s.schedule()
    retire(s, third, [("a", 4, 1)])
    retire(s, second, [("a", 2, 0)])
    assert [(r["producer_plan_id"], r["producer_schedule_id"], r["applied_k"], r["worker_k"])
            for r in events(path, "verification_feedback")] == [(2, 2, 4, 4), (1, 1, 2, 2)]


def test_t_act_008_duplicate_callback_is_skipped(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    s.schedule()
    out = s.schedule()
    retire(s, out, [("a", 3, 0), ("a", 3, 0)])
    c = s._planner.adapter.get_controller("a")
    assert c.verification_sample_count == 1 and c.alpha_ema == pytest.approx(0.64)
    assert len(events(path, "verification_feedback")) == len(events(path, "verification_feedback_skipped")) == 1


def test_t_tel_001_decision_fields_distinguish_policy_and_estimator(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, min_warmup_samples=3)
    add_running(s, make_request("a"))
    s.schedule()
    retire(s, s.schedule(), [("a", 3, 0)])
    s.schedule()
    row = events(path, "planned_decision")[-1]
    assert row["selected_k"] == row["applied_k"] == 3
    assert row["effective_alpha"] == pytest.approx(0) and row["rule_alpha"] == 0.8
    assert row["warmup_active"] and row["verification_sample_count"] == 1
    assert row["request_incarnation"] == s._lineage.active["a"][1] and row["plan_id"] == 3
    assert row["reason"].startswith("High acceptance") and row["memory_state"] == "clear"


def test_t_tel_002_batch_fields_for_pending_recovery(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    set_alpha(s._planner.adapter, "a", 0.7)
    s.schedule()
    set_alpha(s._planner.adapter, "a", 0.95)
    s.schedule()
    row = events(path, "batch_policy")[-1]
    assert {key: row[key] for key in ("plan_id", "fresh_max_k", "recovery_cap_k", "worker_k",
                                     "eligible_batch_size", "pending_target_k", "pending_streak",
                                     "required_hold_steps")} == dict(plan_id=2, fresh_max_k=4,
           recovery_cap_k=2, worker_k=2, eligible_batch_size=1, pending_target_k=4,
           pending_streak=1, required_hold_steps=3)


def test_t_tel_003_feedback_fields_join_to_exact_producer(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    add_running(s, make_request("a"))
    s.schedule()
    retire(s, s.schedule(), [("a", 3, 0)], invalid={"a": 1})
    row = events(path, "verification_feedback")[0]
    assert (row["request_id"], row["request_incarnation"], row["producer_plan_id"],
            row["producer_schedule_id"], row["drafted_tokens"], row["accepted_tokens"]) == (
                "a", s._lineage.active["a"][1], 1, 1, 2, 0)
    assert row["feedback_counted_for_warmup"] is True and row["verification_sample_count"] == 1


def test_t_tel_004_reason_histogram_and_distinct_width_streams(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, history_limit=1)
    add_running(s, make_request("a"), make_request("b"))
    set_alpha(s._planner.adapter, "a", 0.5)
    set_alpha(s._planner.adapter, "b", 0.7)
    s.schedule()
    row = events(path, "planned_decision")[0]
    assert (row["request_id"], row["selected_k"], row["applied_k"]) == ("a", 0, 0)
    assert events(path, "batch_policy")[0]["worker_k"] == 2
    s.schedule()
    final = s._planner.finish_request("a")
    assert final["reason_counts"] == {"fallback": 2}
    assert final["tree_budgets"] == [0]
    assert "batch_k_counts" not in final


def test_replacement_preserves_measured_profile_plumbing(measured_profile, tmp_path):
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(measured_profile), encoding="utf-8")
    p = AdaptiveBudgetPlanner.from_env({
        "ADAPTIVE_REPLACEMENT_POLICY": "1", "FLEXSDS_PROFILE_PATH": str(path),
        "ADAPTIVE_CONTROLLER_KWARGS": '{"min_k": 2, "history_limit": 1}',
    })
    c = p.adapter.get_controller("0")
    assert isinstance(c.cost_profile, EmpiricalCostProfile)
    assert c.candidate_budgets == measured_profile["grid"]["candidate_budgets"]
    assert c.cost_profile.get_cost(2, 1) == measured_profile["cost_matrix_ms"]["k2_b1"]
    assert plan(p, 0.5).worker_k == 2
    assert len(c.get_stats()["history"]) == 1


def test_t_warm_008_reset_alpha_is_consistent():
    c = controller(initial_alpha=0.8, min_warmup_samples=3)
    c.reset(initial_alpha=0.5)
    for _ in range(2):
        decision = c.select_budget()
        assert c.initial_alpha == c.alpha_ema == decision.effective_alpha == decision.rule_alpha == 0.5
        assert decision.warmup_active and decision.verification_sample_count == 0 and decision.k == 0
        feedback(c)
        c.reset()


def test_t_tel_005_mismatched_birth_width_remains_observable(tmp_path):
    from types import SimpleNamespace
    from src.server.lineage import DraftOrigin, Execution, LineageLedger
    p = planner(step_telemetry_path=str(tmp_path / "width.jsonl"))
    ledger = LineageLedger()
    ledger.on_policy_draft = p.record_draft_birth
    origin = DraftOrigin("a", 1, 1, 1, 3, 4)
    output = SimpleNamespace(num_spec_tokens_to_schedule=4)
    ledger.executions[id(output)] = Execution(output, {"a": origin})
    runner = SimpleNamespace(_draft_token_ids=SimpleNamespace(shape=(1, 3)),
                             input_batch=SimpleNamespace(req_ids=["a"]))
    ledger.produced(output, runner)
    row = events(tmp_path / "width.jsonl", "draft_policy_produced")[0]
    assert row["planned_worker_k"] == 4 and row["physical_draft_width"] == 3
    assert origin.physical_draft_width == 3 and ledger.generation is None


def test_t_tel_006_feedback_uses_exact_producer_birth(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path, hold=0)
    add_running(s, make_request("a"))
    set_alpha(s._planner.adapter, "a", 0.7)
    s.schedule()
    verifier = s.schedule()
    origin, _ = s._lineage.get(verifier).consumed["a"]
    assert origin.physical_draft_width == 2
    set_alpha(s._planner.adapter, "a", 0.95)
    s.schedule()
    retire(s, verifier, [("a", 2, 0)])
    row = events(path, "verification_feedback")[0]
    assert row["producer_schedule_id"] == 1
    assert row["planned_worker_k"] == row["physical_draft_width"] == 2


def test_t_life_005_late_cleanup_protects_new_incarnation(scheduler_module, monkeypatch, tmp_path):
    s, path = runtime(scheduler_module, monkeypatch, tmp_path)
    old = make_request("a")
    add_running(s, old)
    s.schedule()
    s.running.remove(old)
    fresh = make_request("a")
    add_running(s, fresh)
    s.schedule()
    state = s._planner.adapter.get_controller("a")
    incarnation = s._lineage.active["a"][1]
    assert s._free_request(old) == (None, None)
    assert s.requests["a"] is fresh and fresh in s.running
    assert s._planner.adapter.get_controller("a") is state
    assert s._lineage.active["a"] == (fresh, incarnation)
    assert s.freed == []
