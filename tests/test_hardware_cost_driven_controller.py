"""Tests for hardware cost-driven AdaptiveDraftController and non-monotonicity guards (R6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def test_inspect_monotonicity_detects_a10g_small_batch_inversions():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    anomalies = inspect_cost_matrix_monotonicity(data["cost_matrix_ms"])
    below_k0_pairs = {(item["batch"], item["k"]) for item in anomalies if item["below_k0"]}
    assert (1, 1) in below_k0_pairs
    assert (2, 1) in below_k0_pairs
    assert (2, 2) in below_k0_pairs
    assert (4, 1) in below_k0_pairs
    assert (4, 2) in below_k0_pairs
    assert all(item["batch"] not in {8, 16, 32} for item in anomalies if item["below_k0"])


def test_inspect_monotonicity_returns_empty_for_strictly_increasing_matrix():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    clean = {
        "k0_b1": 15.0,
        "k1_b1": 16.0,
        "k2_b1": 18.0,
        "k0_b8": 16.0,
        "k1_b8": 20.0,
        "k2_b8": 25.0,
    }
    assert inspect_cost_matrix_monotonicity(clean) == []


def _load_a10g_cost_profile():
    from src.proposer.controller import EmpiricalCostProfile

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


def test_cost_driven_adapts_k_across_batch_sizes_on_real_a10g_profile():
    """AC-01: Cùng alpha=0.70, K giảm dần theo batch_size trên profile A10G thật."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    expected = {1: 4, 8: 3, 16: 2, 32: 0}
    for batch_size, expected_k in expected.items():
        ctrl = AdaptiveDraftController(
            initial_alpha=0.70,
            candidate_budgets=budgets,
            cost_profile=profile,
            min_speedup_ratio=1.05,
        )
        decision = ctrl.select_budget(
            batch_size=batch_size, free_vram_mb=16000.0, kv_cache_usage_pct=10.0
        )
        assert decision.k == expected_k
        assert decision.decision_policy == "cost_model"
        assert decision.cost_k_ms is not None and decision.cost_k_ms > 0
        assert decision.cost_baseline_ms == pytest.approx(profile.get_cost(0, batch_size))
        assert decision.speedup is not None and decision.speedup >= 1.0


def test_cost_driven_changes_k_for_different_hardware_profiles():
    """AC-02: Cùng alpha=0.70 và B=8, hai GPU profile khác nhau cho ra K khác nhau."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    fast_verify = EmpiricalCostProfile(
        matrix={(0, 8): 18.0, (1, 8): 18.5, (2, 8): 20.0, (3, 8): 22.0, (4, 8): 25.0}
    )
    slow_verify = EmpiricalCostProfile(
        matrix={(0, 8): 18.0, (1, 8): 42.0, (2, 8): 56.0, (3, 8): 68.0, (4, 8): 75.0}
    )
    c_fast = AdaptiveDraftController(
        initial_alpha=0.70, candidate_budgets=[0, 1, 2, 3, 4], cost_profile=fast_verify
    )
    c_slow = AdaptiveDraftController(
        initial_alpha=0.70, candidate_budgets=[0, 1, 2, 3, 4], cost_profile=slow_verify
    )
    assert c_fast.select_budget(batch_size=8).k >= 3
    assert c_slow.select_budget(batch_size=8).k == 0


def test_cost_driven_non_monotonic_a10g_falls_back_at_zero_alpha():
    """AC-03 & Review Focus #1: Không bao giờ chọn K>0 ở alpha=0 dù C(1,B) < C(0,B)."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    for batch_size in (1, 2, 4, 8, 16, 32):
        ctrl = AdaptiveDraftController(
            initial_alpha=0.0, candidate_budgets=budgets, cost_profile=profile
        )
        decision = ctrl.select_budget(batch_size=batch_size)
        assert decision.k == 0
        assert decision.status == "fallback_gate"
        assert decision.non_monotonic_cost is (batch_size in {1, 2, 4})


def test_cost_driven_preserves_memory_guard_priority_and_probing():
    """AC-04: Memory guard ưu tiên cao hơn cost model; cost fallback kích hoạt periodic probing."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.95,
        candidate_budgets=budgets,
        cost_profile=profile,
        probe_interval=3,
        probe_k=2,
        memory_cooldown_steps=3,
    )
    mem_dec = ctrl.select_budget(batch_size=1, free_vram_mb=500.0, kv_cache_usage_pct=95.0)
    assert mem_dec.k == 0
    assert mem_dec.status == "memory_guard"
    assert mem_dec.decision_policy == "memory_guard"

    ctrl_probe = AdaptiveDraftController(
        initial_alpha=0.50,
        candidate_budgets=budgets,
        cost_profile=profile,
        probe_interval=3,
        probe_k=2,
    )
    assert ctrl_probe.select_budget(batch_size=32).k == 0
    assert ctrl_probe.select_budget(batch_size=32).k == 0
    probe_dec = ctrl_probe.select_budget(batch_size=32)
    assert probe_dec.k == 2
    assert probe_dec.status == "probing"
    assert probe_dec.decision_policy == "cost_model"


def test_cost_driven_probing_clamps_to_candidate_budgets():
    """Review Focus #3: probe_k=2 bị kẹp về k=1 nếu candidate_budgets chỉ có [0, 1]."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    profile = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): 40.0})
    ctrl = AdaptiveDraftController(
        initial_alpha=0.2,
        candidate_budgets=[0, 1],
        cost_profile=profile,
        probe_interval=2,
        probe_k=2,
    )
    assert ctrl.select_budget(batch_size=1).k == 0
    probe_dec = ctrl.select_budget(batch_size=1)
    assert probe_dec.k == 1
    assert probe_dec.status == "probing"


def test_cost_driven_tie_breaks_toward_smaller_k():
    """Review Focus #4: Khi hai ứng viên có speedup chênh < 1e-6, chọn k nhỏ hơn."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    profile = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): 15.0, (2, 1): 17.5})
    ctrl = AdaptiveDraftController(
        initial_alpha=0.5, candidate_budgets=[0, 1, 2], cost_profile=profile
    )
    assert ctrl.select_budget(batch_size=1).k == 1


def test_cost_driven_rejects_non_positive_or_non_finite_costs():
    """Review Focus #2: Chi phí <= 0 hoặc NaN/inf phải ném ValueError."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    bad = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): float("nan")})
    ctrl = AdaptiveDraftController(
        initial_alpha=0.8, candidate_budgets=[0, 1], cost_profile=bad
    )
    with pytest.raises(ValueError, match="finite"):
        ctrl.select_budget(batch_size=1)


def test_cost_driven_handles_intermediate_batch_size_via_closest_grid_point():
    """Review Focus #5: Batch trung gian B=3 tra lưới A10G gần nhất (B=2) nhất quán."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.70, candidate_budgets=budgets, cost_profile=profile
    )
    d3 = ctrl.select_budget(batch_size=3)
    d2 = AdaptiveDraftController(
        initial_alpha=0.70, candidate_budgets=budgets, cost_profile=profile
    ).select_budget(batch_size=2)
    assert d3.k == d2.k
    assert d3.cost_k_ms == pytest.approx(d2.cost_k_ms)


def test_planner_from_env_records_cost_driven_telemetry_and_warns_non_monotonic(tmp_path, caplog):
    """AC-05: Planner nạp FLEXSDS_PROFILE_PATH, log cảnh báo không đơn điệu và ghi telemetry R6/R8."""
    import logging
    from src.server.budget_planner import AdaptiveBudgetPlanner

    step_path = tmp_path / "serving_steps.jsonl"
    with caplog.at_level(logging.WARNING, logger="src.server.budget_planner"):
        planner = AdaptiveBudgetPlanner.from_env(
            {
                "FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH),
                "ADAPTIVE_STEP_TELEMETRY_PATH": str(step_path),
                "ADAPTIVE_CONTROLLER_KWARGS": json.dumps({"initial_alpha": 0.70}),
            }
        )

    assert "non-monotonic" in caplog.text.lower()

    plan_b1 = planner.plan_step(["req-a"], batch_size=1, kv_usage=0.1, free_vram_mb=16000.0)
    plan_b32 = planner.plan_step(["req-b"], batch_size=32, kv_usage=0.1, free_vram_mb=16000.0)
    assert plan_b1.batch_k == 4
    assert plan_b32.batch_k == 0

    lines = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    rec_b1, rec_b32 = lines[0], lines[1]

    assert rec_b1["decision_policy"] == "cost_model"
    assert rec_b1["hardware_profile"]["device"] == "NVIDIA A10"
    assert rec_b1["hardware_profile"]["sm_architecture"] == "sm_86"
    assert rec_b1["hardware_profile"]["sha256"]
    assert rec_b1["theoretical_speedup"] > 1.05
    assert rec_b1["dynamic_alpha_star"] is not None
    assert rec_b1["cost_k_ms"] > 0
    assert rec_b1["cost_baseline_ms"] > 0
    assert rec_b1["cost_ref_ms"] > 0
    assert rec_b1["non_monotonic_cost"] is True

    assert rec_b32["decision_policy"] == "cost_model"
    assert rec_b32["scalar_runtime_budget"] == 0
    assert rec_b32["non_monotonic_cost"] is False


def test_planner_max_k_ceiling_excludes_unexecutable_k_from_cost_model_search(tmp_path):
    """Adversarial check: khi max_k=3 tại B=32, alpha=0.90 (k=4 lời nhưng k=3 lỗ),
    planner phải loại k=4 khỏi tập ứng viên và chọn k=0 thay vì chọn k=4 rồi kẹp xuống k=3."""
    from src.server.budget_planner import AdaptiveBudgetPlanner

    step_path = tmp_path / "serving_steps.jsonl"
    planner = AdaptiveBudgetPlanner.from_env(
        {
            "FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH),
            "ADAPTIVE_MAX_K": "4",
            "ADAPTIVE_STEP_TELEMETRY_PATH": str(step_path),
            "ADAPTIVE_CONTROLLER_KWARGS": json.dumps({"initial_alpha": 0.90}),
        }
    )
    # Giả lập AdaptiveAsyncScheduler.__init__ thu hẹp max_k theo num_spec_tokens = 3
    planner.max_k = 3
    plan = planner.plan_step(["req-b32"], batch_size=32, kv_usage=0.1, free_vram_mb=16000.0)
    assert plan.batch_k == 0
    assert plan.per_request_k["req-b32"] == 0

    rec = json.loads(step_path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["decision_policy"] == "cost_model"
    assert rec["scalar_runtime_budget"] == 0
    assert rec["cost_k_ms"] == pytest.approx(rec["cost_baseline_ms"])


def test_controller_rejects_boolean_speedup_ratio_and_cost_entries():
    """Adversarial check: bool là lớp con của int trong Python, không được lọt qua kiểm tra số thực."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    with pytest.raises(ValueError, match="min_speedup_ratio"):
        AdaptiveDraftController(min_speedup_ratio=True)

    bool_cost_profile = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): True})
    ctrl = AdaptiveDraftController(
        initial_alpha=0.8, candidate_budgets=[0, 1], cost_profile=bool_cost_profile
    )
    with pytest.raises(ValueError, match="finite and positive"):
        ctrl.select_budget(batch_size=1)


def test_cost_driven_handles_nan_alpha_safely_without_crashing():
    """Adversarial check: alpha=NaN fallback an toàn về k=0 và không sinh giá trị NaN."""
    import math
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    ctrl = AdaptiveDraftController(initial_alpha=0.8, candidate_budgets=budgets, cost_profile=profile)
    decision = ctrl.apply_cost_driven_budget_rule(alpha=float("nan"), batch_size=1)
    assert decision.k == 0
    assert math.isfinite(decision.alpha_ema)
    assert decision.speedup == 1.0


def test_tree_adaptive_decision_does_not_inherit_threshold_policy():
    """Adversarial check: TreeAdaptiveDecision không bị kế thừa nhầm decision_policy='threshold'."""
    from src.proposer.tree_adaptive import TreeAdaptiveDecision

    dec = TreeAdaptiveDecision(k=4, alpha_ema=0.8, expected_tokens=2.5, reason="tree")
    assert dec.decision_policy is None
