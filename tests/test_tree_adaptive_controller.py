"""Unit tests and benchmarks for TreeAdaptiveController v0.2."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock

try:
    from flexsds.core.cost_model import FlexCostModel, SolverResult
    from flexsds.core.tree_topology import get_tree_topology
except ModuleNotFoundError:
    from src.flexsds.core.cost_model import FlexCostModel, SolverResult
    from src.flexsds.core.tree_topology import get_tree_topology

from src.proposer.tree_adaptive import (
    TreeAdaptiveController,
    TreeAdaptiveDecision,
)
from src.proposer.controller import BudgetDecision
from src.proposer.adapter import AdaptiveProposerAdapter


def test_tree_adaptive_controller_initialization():
    """Verifies default and custom initialization parameters."""
    controller = TreeAdaptiveController()
    assert controller.candidate_topologies == ["T0", "T4", "T8", "T16", "T32"]
    assert controller.min_free_vram_mb == 1500.0
    assert controller.kv_cache_high_pct == 90.0
    assert controller.kv_cache_recover_pct == 80.0
    assert controller.cooldown_steps == 3
    assert controller.ema_beta == 0.8
    assert controller.initial_alpha == 0.8

    custom = TreeAdaptiveController(
        candidate_topologies=["T0", "T4", "T8"],
        min_free_vram_mb=2000.0,
        cooldown_steps=5,
    )
    assert custom.candidate_topologies == ["T0", "T4", "T8"]
    assert custom.min_free_vram_mb == 2000.0
    assert custom.cooldown_steps == 5


def test_unbiased_ema_alpha_tracking():
    """Verifies EMA calculation with warm-up bias correction."""
    controller = TreeAdaptiveController(ema_beta=0.8, initial_alpha=0.8)
    assert controller.get_effective_alpha() == 0.8

    # Step 1: observation 1.0
    controller.update(num_accepted=4, num_drafted=4)
    # raw EMA: 0.8*0.8 + 0.2*1.0 = 0.84
    # unbiased: (0.84 - 0.8 * 0.8^1) / (1 - 0.8^1) = (0.84 - 0.64) / 0.2 = 1.0
    eff = controller.get_effective_alpha()
    assert pytest.approx(eff, abs=1e-5) == 1.0

    # Step 2: observation 0.5
    controller.update(num_accepted=2, num_drafted=4)
    # raw EMA: 0.8*0.84 + 0.2*0.5 = 0.672 + 0.10 = 0.772
    # unbiased: (0.772 - 0.8 * 0.8^2) / (1 - 0.8^2) = (0.772 - 0.512) / 0.36 = 0.26 / 0.36 ≈ 0.7222
    assert pytest.approx(controller.get_effective_alpha(), abs=1e-3) == (0.772 - 0.512) / 0.36


def test_depth_normalized_alpha_update():
    """Verifies that alpha_step normalizes by tree depth d, avoiding collapse on 2D trees."""
    controller = TreeAdaptiveController()
    # Mock last selected topology as T16 (N=16, d=4)
    controller._last_selected_depth = 4
    controller._last_selected_topology = "T16"

    # Perfect acceptance along optimal path: 4 tokens accepted out of 16 drafted
    controller.update(num_accepted=4, num_drafted=16)
    # Should evaluate step alpha as min(1.0, 4 / 4) = 1.0, NOT 4 / 16 = 0.25!
    assert controller._window[-1] == 1.0

    # 2 tokens accepted out of 16 drafted
    controller.update(num_accepted=2, num_drafted=16)
    assert controller._window[-1] == 0.5

    # If caller already passes depth (num_drafted <= d), preserve standard ratio
    controller._last_selected_depth = 4
    controller.update(num_accepted=3, num_drafted=4)
    assert controller._window[-1] == 0.75


def test_dual_tier_memory_guard_vram_trigger():
    """Verifies immediate fallback to T0 when free VRAM drops below threshold."""
    controller = TreeAdaptiveController(min_free_vram_mb=1500.0)

    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=1450.0,  # below 1500 MB
        kv_cache_usage_pct=50.0,
    )

    assert isinstance(decision, BudgetDecision)
    assert isinstance(decision, TreeAdaptiveDecision)
    assert decision.topology_name == "T0"
    assert decision.num_nodes == 0
    assert decision.k == 0
    assert decision.status == "memory_guard"
    assert "VRAM" in decision.reason
    assert controller._memory_guard_active is True


def test_dual_tier_memory_guard_kv_cache_trigger():
    """Verifies immediate fallback to T0 when KV cache exceeds 90%."""
    controller = TreeAdaptiveController(kv_cache_high_pct=90.0)

    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=16000.0,
        kv_cache_usage_pct=91.5,  # above 90%
    )

    assert decision.topology_name == "T0"
    assert decision.num_nodes == 0
    assert decision.k == 0
    assert decision.status == "memory_guard"
    assert "KV" in decision.reason
    assert controller._memory_guard_active is True


def test_memory_guard_cooldown_and_recovery_conditions():
    """Verifies 3-step cooldown and dual condition for recovery (KV <= 80% AND VRAM >= 1500MB)."""
    controller = TreeAdaptiveController(
        min_free_vram_mb=1500.0,
        kv_cache_high_pct=90.0,
        kv_cache_recover_pct=80.0,
        cooldown_steps=3,
    )

    # 1. Trigger guard
    controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=92.0)
    assert controller._memory_guard_active is True

    # 2. Conditions return to normal (KV=75%, VRAM=16000), but cooldown must hold 3 steps
    # Step 1
    d1 = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=75.0)
    assert d1.topology_name == "T0"
    assert d1.status == "memory_guard"

    # Step 2
    d2 = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=75.0)
    assert d2.topology_name == "T0"
    assert d2.status == "memory_guard"

    # Step 3
    d3 = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=75.0)
    assert d3.topology_name == "T0"
    assert d3.status == "memory_guard"

    # Step 4: Cooldown complete, conditions healthy -> should recover
    d4 = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=75.0)
    assert controller._memory_guard_active is False
    assert d4.status != "memory_guard"

    # Case: KV <= 80% but VRAM < 1500MB -> Must NOT recover
    controller.select_budget(batch_size=1, free_vram_mb=1000.0, kv_cache_usage_pct=50.0)
    for _ in range(5):
        d_bad_vram = controller.select_budget(batch_size=1, free_vram_mb=1400.0, kv_cache_usage_pct=50.0)
        assert d_bad_vram.topology_name == "T0"
        assert d_bad_vram.status == "memory_guard"


def test_no_probing_under_memory_guard():
    """Verifies that probing is strictly forbidden when Memory Guard is active."""
    controller = TreeAdaptiveController(min_free_vram_mb=1500.0)

    # Set fallback steps high enough to normally trigger probing
    controller._fallback_steps = 100

    # Call under VRAM starvation
    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=1200.0,
        kv_cache_usage_pct=50.0,
    )

    assert decision.topology_name == "T0"
    assert decision.status == "memory_guard"
    # Probe counter must be reset
    assert controller._fallback_steps == 0


def test_concurrency_throttling_limits():
    """Verifies topology cap based on batch size B and cutoff at B > 24 when alpha < 0.85."""
    controller = TreeAdaptiveController(initial_alpha=0.95)

    # B <= 4: Allows up to T32
    d_b4 = controller.select_budget(batch_size=4, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert d_b4.topology_name in ["T0", "T4", "T8", "T16", "T32"]

    # 4 < B <= 12 (e.g. B=8): Max T16
    d_b8 = controller.select_budget(batch_size=8, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert d_b8.topology_name in ["T0", "T4", "T8", "T16"]

    # 12 < B <= 24 (e.g. B=16): Max T8
    d_b16 = controller.select_budget(batch_size=16, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert d_b16.topology_name in ["T0", "T4", "T8"]

    # B > 24 with alpha < 0.85: Throttled to T0 with status="concurrency_throttled"
    controller_low_alpha = TreeAdaptiveController(initial_alpha=0.75)
    d_b32 = controller_low_alpha.select_budget(batch_size=32, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert d_b32.topology_name == "T0"
    assert d_b32.num_nodes == 0
    assert d_b32.status == "concurrency_throttled"


def test_dynamic_fallback_gate_and_adaptive_probing_cycle():
    """Verifies Fallback Gate to T0 when Speedup <= 1.0 and probing intervals."""
    # Mock cost model returning speedup <= 1.0
    mock_cost_model = MagicMock(spec=FlexCostModel)
    mock_cost_model.select_optimal_topology.return_value = ("T0", 1.0)
    mock_cost_model.solve_alpha_star.return_value = SolverResult(
        alpha_star=0.75,
        is_feasible=True,
        status="converged",
        cost_ratio=1.2,
        residual=1e-15,
        iterations=50,
        topology_name="T4",
        batch_size=1,
    )

    controller = TreeAdaptiveController(cost_model=mock_cost_model)

    # For B=1, T_probe = 15 steps
    for step in range(1, 15):
        d = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
        assert d.topology_name == "T0", f"Step {step} should be T0"
        assert d.status == "fallback_gate"

    # Step 15: Probe T4
    d_probe = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert d_probe.topology_name == "T4"
    assert d_probe.num_nodes == 4
    assert d_probe.status == "probing"
    assert controller._fallback_steps == 0


def test_zero_crash_resilience():
    """Verifies that internal exceptions safely fall back to T0 without crashing."""
    mock_cost_model = MagicMock(spec=FlexCostModel)
    mock_cost_model.select_optimal_topology.side_effect = RuntimeError("GPU Kernel crashed")

    controller = TreeAdaptiveController(cost_model=mock_cost_model)

    decision = controller.select_budget(batch_size=1, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
    assert decision.topology_name == "T0"
    assert decision.num_nodes == 0
    assert decision.k == 0
    assert decision.status == "zero_crash_fallback"
    assert "Zero-crash fallback" in decision.reason


def test_decision_latency_benchmark():
    """Verifies that select_budget executes well within the 0.05 ms (50 microsecond) SLA."""
    controller = TreeAdaptiveController(initial_alpha=0.85)

    # Warmup
    for _ in range(100):
        controller.select_budget(batch_size=4, free_vram_mb=16000.0, kv_cache_usage_pct=20.0)

    # Best of 5 rounds (as timeit does): one round slowed by other processes on a busy machine
    # must not fail the SLA check, while a genuinely slow select_budget still fails every round.
    iterations = 10000
    rounds = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            _ = controller.select_budget(batch_size=4, free_vram_mb=16000.0, kv_cache_usage_pct=20.0)
        rounds.append(time.perf_counter() - start)
    avg_latency_ms = (min(rounds) / iterations) * 1000.0

    print(f"\nAverage decision latency: {avg_latency_ms * 1000:.3f} microseconds ({avg_latency_ms:.6f} ms)")
    assert avg_latency_ms < 0.05, f"Decision latency too high: {avg_latency_ms:.6f} ms >= 0.05 ms"


def test_proposer_adapter_integration():
    """Verifies end-to-end proposal flow through AdaptiveProposerAdapter."""
    adapter = AdaptiveProposerAdapter(
        controller_cls=TreeAdaptiveController,
        controller_kwargs={"initial_alpha": 0.85},
    )

    req_id = "req_tree_test_001"
    mock_generator = MagicMock(return_value=[101, 102, 103, 104])

    proposal = adapter.propose(
        request_id=req_id,
        draft_generator=mock_generator,
        batch_size=1,
        free_vram_mb=16000.0,
        kv_cache_usage_pct=20.0,
    )

    # Controller should choose an active tree
    assert proposal.budget_k > 0
    assert proposal.skipped_draft is False
    assert len(proposal.draft_tokens) == 4

    # Feedback update
    new_alpha = adapter.on_verification_completed(
        request_id=req_id,
        num_accepted=3,
        num_drafted=proposal.budget_k,
    )
    assert 0.0 <= new_alpha <= 1.0


def test_tree_adaptive_controller_reset():
    """Verifies that reset() cleans up all internal state without leakage across sequences."""
    controller = TreeAdaptiveController(initial_alpha=0.8)
    controller.update(num_accepted=4, num_drafted=4)
    controller.select_budget(batch_size=1, free_vram_mb=1000.0, kv_cache_usage_pct=50.0)
    assert controller._memory_guard_active is True
    assert controller._step_count > 0

    controller.reset(initial_alpha=0.9)
    assert controller.initial_alpha == 0.9
    assert controller.alpha_ema == 0.9
    assert controller._step_count == 0
    assert controller._fallback_steps == 0
    assert controller._memory_guard_active is False
    assert controller._cooldown_counter == 0
    assert controller._guard_reason == ""
    assert controller._last_selected_depth == 0
    assert controller._last_selected_topology == "T0"


def test_adapter_telemetry_forwards_topologies():
    """Verifies that adapter.get_telemetry() includes the topologies sequence."""
    adapter = AdaptiveProposerAdapter(
        controller_cls=TreeAdaptiveController,
        controller_kwargs={"initial_alpha": 0.85},
    )
    req_id = "req_tree_test_telemetry"
    mock_gen = MagicMock(return_value=[1, 2, 3, 4])
    adapter.propose(req_id, mock_gen, batch_size=1)
    telemetry = adapter.get_telemetry(req_id)

    assert "topologies" in telemetry
    assert len(telemetry["topologies"]) >= 1


def test_memory_guard_sentinel_vram_and_recovery_safety():
    """Verifies that sentinel VRAM values (e.g. 1e9) tighten KV-cache guards and prevent premature recovery."""
    ctrl = TreeAdaptiveController(min_free_vram_mb=1500.0, kv_cache_high_pct=90.0, kv_cache_recover_pct=80.0)
    # Case A: VRAM sentinel 1e9 but KV high (78% > tight threshold 75%) -> guard triggers
    dec = ctrl.select_budget(batch_size=1, free_vram_mb=1.0e9, kv_cache_usage_pct=78.0)
    assert dec.topology_name == "T0"
    assert dec.status == "memory_guard"

    # Case B: Starved by physical VRAM then telemetry flaps to 1e9 -> must hold guard
    ctrl.reset()
    ctrl.select_budget(batch_size=1, free_vram_mb=1000.0, kv_cache_usage_pct=50.0)
    assert ctrl._memory_guard_active is True
    # Telemetry flaps to sentinel 1e9 with low KV (50%) -> should NOT prematurely recover
    dec2 = ctrl.select_budget(batch_size=1, free_vram_mb=1.0e9, kv_cache_usage_pct=50.0)
    assert dec2.topology_name == "T0"
    assert dec2.status == "memory_guard"


def test_alpha_observation_count_and_nan_self_healing():
    """Verifies that unobserved steps do not decay EMA debiasing and NaN values are self-healed."""
    ctrl = TreeAdaptiveController(initial_alpha=0.8, ema_beta=0.8)
    # Cold-start or T0 steps with num_drafted == 0
    ctrl.update(num_accepted=0, num_drafted=0)
    assert ctrl._step_count == 1
    assert ctrl._observation_count == 0
    assert ctrl.get_effective_alpha() == 0.8  # No decay of beta_power

    # Valid step
    ctrl.update(num_accepted=3, num_drafted=4)
    assert ctrl._observation_count == 1
    assert 0.0 <= ctrl.get_effective_alpha() <= 1.0

    # Inject NaN into alpha_ema
    import math
    ctrl.alpha_ema = float("nan")
    alpha_recovered = ctrl.get_effective_alpha()
    assert not math.isnan(alpha_recovered)
    assert alpha_recovered == pytest.approx(0.8)
    assert not math.isnan(ctrl.alpha_ema)  # Self-healed


def test_concurrency_throttling_and_probing_safety():
    """Verifies that high concurrency throttling never probes while memory guard is active."""
    ctrl = TreeAdaptiveController()
    ctrl.alpha_ema = 0.5
    ctrl._observation_count = 10
    # Simulate steps up to probing interval (30) under memory pressure
    for _ in range(35):
        dec = ctrl.select_budget(batch_size=32, free_vram_mb=500.0, kv_cache_usage_pct=95.0)
        assert dec.topology_name == "T0"
        assert dec.status == "memory_guard"  # Never probe under memory guard!


def test_adapter_proposal_output_carries_topology_name():
    """Verifies that ProposalOutput exposes topology_name for TreeAdaptiveController and None for linear."""
    from src.proposer.adapter import AdaptiveProposerAdapter
    from src.proposer.tree_adaptive import TreeAdaptiveController
    from src.proposer.controller import AdaptiveDraftController

    # With TreeAdaptiveController
    tree_adapter = AdaptiveProposerAdapter(controller_cls=TreeAdaptiveController)
    out_tree = tree_adapter.propose("req_tree", lambda k: [1] * k, batch_size=1)
    assert hasattr(out_tree, "topology_name")
    assert out_tree.topology_name in ["T0", "T4", "T8", "T16", "T32"]

    # With legacy linear AdaptiveDraftController
    linear_adapter = AdaptiveProposerAdapter(controller_cls=AdaptiveDraftController)
    out_linear = linear_adapter.propose("req_linear", lambda k: [1] * k, batch_size=1)
    assert hasattr(out_linear, "topology_name")
    assert out_linear.topology_name is None or out_linear.topology_name == "T0"


def test_fallback_gate_vs_zero_crash_exception():
    """Verifies clear separation between economic infeasibility (fallback_gate) and cost model crashes (zero_crash_fallback)."""
    ctrl = TreeAdaptiveController()
    # Path A: Infeasible speedup (force mock or candidate with speedup <= 1.0)
    # Controller returns fallback_gate or ok, not crashing
    dec = ctrl.select_budget(batch_size=1)
    assert dec.status in ["ok", "fallback_gate"]

    # Path B: Exception injection in cost model
    class FaultyCostModel:
        def select_optimal_topology(self, *args, **kwargs):
            raise RuntimeError("CUDA device communication failure")
        def solve_alpha_star(self, *args, **kwargs):
            raise RuntimeError("Cost model solver failure")

    faulty_ctrl = TreeAdaptiveController(cost_model=FaultyCostModel())
    dec_crash = faulty_ctrl.select_budget(batch_size=1)
    assert dec_crash.topology_name == "T0"
    assert dec_crash.status == "zero_crash_fallback"
    assert "Zero-crash fallback" in dec_crash.reason


def test_tree_adaptive_controller_reset_without_args_clears_alpha_ema():
    """Critical 1: Verifies reset() without args resets alpha_ema to initial_alpha to avoid false fallback gate."""
    ctrl = TreeAdaptiveController(initial_alpha=0.8, ema_beta=0.8)
    # Simulate a degraded sequence
    ctrl.update(num_accepted=0, num_drafted=4)
    ctrl.update(num_accepted=0, num_drafted=4)
    ctrl.update(num_accepted=0, num_drafted=4)
    assert ctrl.alpha_ema < 0.5

    # Reset without args (standard session recycling)
    ctrl.reset()
    assert ctrl.alpha_ema == 0.8  # Must not leak degraded alpha
    assert ctrl._observation_count == 0

    # New sequence has 100% acceptance
    ctrl.update(num_accepted=4, num_drafted=4)
    eff_alpha = ctrl.get_effective_alpha()
    # Should be close to 1.0, NOT 0.0 (false fallback gate)
    assert eff_alpha >= 0.9, f"False fallback gate trigger: eff_alpha={eff_alpha} < 0.9"


def test_invalid_kv_cache_telemetry_triggers_memory_guard():
    """Important 2: Verifies that NaN, None, < 0, and > 100 KV telemetry trigger Memory Guard defensively."""
    ctrl = TreeAdaptiveController()
    invalid_kvs = [None, float("nan"), -5.0, 105.0]
    for bad_kv in invalid_kvs:
        dec = ctrl.select_budget(batch_size=1, free_vram_mb=8000.0, kv_cache_usage_pct=bad_kv)
        assert dec.topology_name == "T0", f"Failed for bad KV: {bad_kv}"
        assert dec.status == "memory_guard"
        assert "invalid KV telemetry" in dec.reason or "Memory guard active" in dec.reason







