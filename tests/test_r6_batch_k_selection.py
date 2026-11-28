"""Batch-level K selection and feasible-only batch probing for R6 (SPEC-R6-BATCHK-20261124)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile
from src.server.budget_planner import AdaptiveBudgetPlanner

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _profile():
    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


def _expected_tokens(k: int, alpha: float) -> float:
    return 1.0 + sum(alpha**j for j in range(1, k + 1))


def _best_k_by_throughput(profile, budgets, alphas, batch_size, ratio=1.05):
    """Independent reference: argmax_K sum_i E[tau_i(K)] / C(K, B), K>0 only if it beats K=0 by `ratio`.

    K=0 is costed at C_ref(B) = min_K C(K, B), as in the per-request rule (some C(1, B) cells are below C(0, B)).
    """
    costs = {k: profile.get_cost(k, batch_size) for k in budgets}
    scores = {k: sum(_expected_tokens(k, a) for a in alphas) / costs[k] for k in budgets}
    scores[0] = len(alphas) / min(costs.values())
    best = 0
    for k in sorted(budgets):
        if k > 0 and scores[k] >= ratio * scores[0] and scores[k] > scores[best] + 1e-9:
            best = k
    return best


def make_planner(k_selection="batch", *, controller_overrides=None, **planner_kwargs):
    profile, budgets = _profile()
    kwargs = {
        "cost_profile": profile,
        "candidate_budgets": list(budgets),
        "alpha_estimator": "geometric_mle",
        "enable_vram_guard": False,
    }
    kwargs.update(controller_overrides or {})
    return AdaptiveBudgetPlanner(
        adapter=AdaptiveProposerAdapter(controller_kwargs=kwargs), k_selection=k_selection, **planner_kwargs
    )


def set_alpha(planner, request_id, alpha):
    ctrl = planner.adapter.get_controller(request_id)
    ctrl._mle_accepted = alpha * 100.0
    ctrl._mle_tested = 100.0


def ids(n):
    return [f"r{i}" for i in range(n)]


def test_batch_mode_chooses_one_k_for_mixed_alpha():
    """AC-01: B=8, mixed alpha, one K for the batch and no per-request trimming."""
    planner = make_planner()
    alphas = [0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3]
    for rid, a in zip(ids(8), alphas):
        set_alpha(planner, rid, a)
    plan = planner.plan_step(ids(8), batch_size=8, kv_usage=0.10)

    assert len(set(plan.per_request_k.values())) == 1
    assert plan.batch_k == next(iter(plan.per_request_k.values()))
    assert all(k == plan.batch_k for k in plan.per_request_k.values())

    profile, budgets = _profile()
    assert plan.batch_k == _best_k_by_throughput(profile, [b for b in budgets if b <= 4], alphas, 8)


def test_batch_mode_turns_drafting_off_at_b32_and_stays_on_at_b1():
    """AC-02: at B=32 the measured alpha (0.68) cannot pay for any K; at B=1 K is 3 or 4."""
    planner = make_planner()
    for rid in ids(32):
        set_alpha(planner, rid, 0.68)
    plan = planner.plan_step(ids(32), batch_size=32, kv_usage=0.10)
    assert plan.batch_k == 0
    assert set(plan.per_request_k.values()) == {0}

    single = make_planner()
    set_alpha(single, "r0", 0.6)
    single_plan = single.plan_step(["r0"], batch_size=1, kv_usage=0.10)
    assert single_plan.batch_k in {3, 4}


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 8, 16, 32])
def test_uniform_alpha_matches_legacy_rule_on_noisy_profile_cells(batch_size):
    """With one shared alpha, batch mode equals the per-request rule, also where C(1, B) < C(0, B) (B <= 4)."""
    for alpha in (0.0, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        legacy, batch = make_planner("max_k", controller_overrides={"probe_interval": 0}), make_planner("batch")
        for planner in (legacy, batch):
            for rid in ids(batch_size):
                set_alpha(planner, rid, alpha)
        expected = legacy.plan_step(ids(batch_size), batch_size=batch_size, kv_usage=0.10).batch_k
        got = batch.plan_step(ids(batch_size), batch_size=batch_size, kv_usage=0.10).batch_k
        assert got == expected, f"alpha={alpha}"


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
def test_b1_matches_legacy_selection(alpha):
    """AC-03: with one request, batch mode picks the same K as the current per-request selection."""
    legacy = make_planner("max_k")
    batch = make_planner("batch")
    for planner in (legacy, batch):
        set_alpha(planner, "r0", alpha)
    assert (
        batch.plan_step(["r0"], batch_size=1, kv_usage=0.10).batch_k
        == legacy.plan_step(["r0"], batch_size=1, kv_usage=0.10).batch_k
    )


def test_no_per_request_probing_at_high_load():
    """AC-04 (high load): at B=32 no probe can pay off, so 200 steps at K=0 stay at K=0."""
    planner = make_planner()
    for rid in ids(32):
        set_alpha(planner, rid, 0.68)
    for _ in range(200):
        plan = planner.plan_step(ids(32), batch_size=32, kv_usage=0.10)
        assert plan.batch_k == 0
        assert all(k == 0 for k in plan.per_request_k.values())


def test_batch_probes_rarely_with_the_cheapest_feasible_k_at_low_load():
    """AC-04 (low load): below break-even at B=8 (alpha 0.02) the batch probes once per probe_interval steps."""
    planner = make_planner()
    for rid in ids(8):
        set_alpha(planner, rid, 0.02)
    probe_steps = []
    for step in range(200):
        plan = planner.plan_step(ids(8), batch_size=8, kv_usage=0.10)
        if plan.batch_k > 0:
            probe_steps.append((step, plan.batch_k))
    assert [step for step, _ in probe_steps] == list(range(19, 200, 20))  # one probe per 20 batch steps
    assert {k for _, k in probe_steps} == {1}  # K=1 has the smallest break-even alpha at B=8
    profile, _ = _profile()
    ctrl = AdaptiveDraftController(cost_profile=profile)
    for _, k in probe_steps:
        star = ctrl.solve_break_even_alpha(k, profile.get_cost(k, 8) / profile.get_cost(0, 8))
        assert star is not None and star <= 0.80


def test_probing_is_off_in_controllers_and_counted_per_batch():
    """AC-05: batch mode never emits a per-request 'Periodic probe' decision."""
    planner = make_planner()
    for rid in ids(8):
        set_alpha(planner, rid, 0.02)
    reasons = set()
    for _ in range(60):
        plan = planner.plan_step(ids(8), batch_size=8, kv_usage=0.10)
        reasons |= {decision.reason for decision in plan.per_request_decisions.values()}
    assert planner.adapter.controller_kwargs["probe_interval"] == 0
    assert not any("Periodic probe" in r for r in reasons)


def test_memory_guard_still_limits_k():
    """AC-06: the memory guard keeps its priority; nothing above its K reaches the batch."""
    planner = make_planner(controller_overrides={"enable_vram_guard": True, "min_free_vram_mb": 5000.0})
    for rid in ids(4):
        set_alpha(planner, rid, 0.9)
    plan = planner.plan_step(ids(4), batch_size=4, kv_usage=0.10, free_vram_mb=100.0)
    assert plan.batch_k == 0
    assert set(plan.per_request_k.values()) == {0}


def test_batch_telemetry_fields_are_additive(tmp_path):
    """AC-07: planned_decision carries the batch decision; existing fields are unchanged."""
    steps = tmp_path / "steps.jsonl"
    planner = make_planner(step_telemetry_path=str(steps), step_telemetry_flush_every=1)
    for rid in ids(8):
        set_alpha(planner, rid, 0.6)
    plan = planner.plan_step(ids(8), batch_size=8, kv_usage=0.10)
    planner.close()
    record = [json.loads(line) for line in steps.read_text(encoding="utf-8").splitlines()][-1]
    assert record["batch_k_selected"] == plan.batch_k
    assert record["batch_probe"] is False
    assert record["batch_score"] > 0 and record["batch_score_k0"] > 0
    for field in ("effective_alpha", "alpha_estimator", "decision_reason", "theoretical_speedup", "batch_scalar_budget"):
        assert field in record


def test_max_k_mode_keeps_the_current_per_request_behaviour():
    """AC-08: 'max_k' (the constructor default) keeps max(k_i) with per-request trimming."""
    planner = make_planner("max_k")
    assert AdaptiveBudgetPlanner().k_selection == "max_k"
    set_alpha(planner, "high", 0.8)
    set_alpha(planner, "low", 0.3)
    plan = planner.plan_step(["high", "low"], batch_size=2, kv_usage=0.10)
    assert plan.batch_k == max(plan.per_request_k.values())
    assert plan.per_request_k["high"] >= plan.per_request_k["low"]


def test_invalid_selection_mode_is_rejected():
    with pytest.raises(ValueError):
        make_planner("per_token")


def test_from_env_reads_and_validates_the_mode(monkeypatch):
    base = {"FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH)}
    assert AdaptiveBudgetPlanner.from_env(base).k_selection == "max_k"
    assert AdaptiveBudgetPlanner.from_env({**base, "ADAPTIVE_K_SELECTION": "batch"}).k_selection == "batch"
    with pytest.raises(ValueError):
        AdaptiveBudgetPlanner.from_env({**base, "ADAPTIVE_K_SELECTION": "auto"})


def _serving_cfg(**extra):
    return {
        "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "target_revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641",
        "draft_revision": "d9016c3860f42cc3f33a87fa5d346fee768f6290",
        "draft_method": "eagle3",
        "profile_path": str(A10G_PROFILE_PATH),
        "enable_adaptive": True,
        "max_num_seqs": 32,
        **extra,
    }


def test_local_runner_turns_batch_selection_on_and_accepts_an_override():
    from flexsds.server.local_runner import LocalServingRunner

    runner = LocalServingRunner()
    assert runner.setup_environment(_serving_cfg())["ADAPTIVE_K_SELECTION"] == "batch"
    assert runner.setup_environment(_serving_cfg(k_selection="max_k"))["ADAPTIVE_K_SELECTION"] == "max_k"
    with pytest.raises(ValueError):
        runner.setup_environment(_serving_cfg(k_selection="per_token"))


def test_non_adaptive_environment_drops_a_leaked_mode(monkeypatch):
    from flexsds.server.local_runner import LocalServingRunner

    monkeypatch.setenv("ADAPTIVE_K_SELECTION", "batch")
    env = LocalServingRunner().setup_environment({"enable_adaptive": False, "static_k": 3})
    assert "ADAPTIVE_K_SELECTION" not in env


def test_modal_runner_forwards_the_mode():
    from flexsds.server.modal_runner import ModalServingRunner

    runner = ModalServingRunner.__new__(ModalServingRunner)

    class _Cred:
        path = Path("/nonexistent/flexsds/config.json")

    runner.cred_mgr = _Cred()
    env = runner.setup_environment({"k_selection": "max_k", "target_model": "t"})
    assert json.loads(env["FLEXSDS_SERVING_CONFIG"])["k_selection"] == "max_k"


def test_runner_and_planner_share_the_same_modes():
    """The runner validates the mode with the same set the planner accepts."""
    from flexsds.server import local_runner
    from src.server import budget_planner

    assert local_runner.K_SELECTION_MODES == budget_planner.K_SELECTION_MODES


def test_batch_mode_falls_back_when_the_controller_has_no_decision_alpha():
    """Another controller type (no decision_alpha, or no cost table) keeps the per-request path."""
    from types import SimpleNamespace

    planner = make_planner("batch")
    profile, budgets = _profile()
    legacy = SimpleNamespace(cost_profile=profile, candidate_budgets=list(budgets), min_speedup_ratio=1.05)
    planner.adapter._active_sessions["r0"] = legacy
    assert planner._select_batch_k({"r0": SimpleNamespace(decision_policy="cost_model")}, 1) is None

    planner.adapter._active_sessions["r0"] = SimpleNamespace(cost_profile=None, candidate_budgets=list(budgets),
                                                              decision_alpha=lambda: (0.6, "geometric_mle"))
    assert planner._select_batch_k({"r0": SimpleNamespace(decision_policy="cost_model")}, 1) is None


def test_batch_mode_without_a_cost_table_keeps_per_request_probing():
    """No cost matrix means no batch selection, so the controllers' own probing must stay on."""
    planner = AdaptiveBudgetPlanner(
        adapter=AdaptiveProposerAdapter(controller_kwargs={"probe_interval": 20}), k_selection="batch"
    )
    assert planner.adapter.controller_kwargs["probe_interval"] == 20
