
"""Regression tests for the current low-acceptance fallback behavior.

Target commit: ddf24d194e13b08b5033109d192184f7f55518d5

These tests document current behavior, NOT an implemented recovery policy.
No vLLM server or GPU is required.
"""

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import BudgetController, CostProfile


def test_low_alpha_fallback_remains_stuck_without_feedback():
    """A request cannot leave fallback without a new alpha observation."""
    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
    )

    decisions = [controller.select_budget() for _ in range(20)]

    assert all(decision.k == 0 for decision in decisions)
    assert controller.get_rolling_alpha() == pytest.approx(0.60)

    # Twenty budget decisions do not imply twenty acceptance observations.
    assert controller.get_stats()["step_count"] == 0
    assert controller.get_stats()["tree_budgets"] == [0] * 20


def test_zero_draft_update_does_not_restore_alpha():
    """update(0, 0) reuses the current EMA, so it cannot cause recovery."""
    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
    )

    for _ in range(20):
        assert controller.select_budget().k == 0
        controller.update(num_accepted=0, num_drafted=0)

    assert controller.get_rolling_alpha() == pytest.approx(0.60)
    assert controller.select_budget().k == 0


def test_adapter_does_not_generate_drafts_during_fallback():
    """The adapter skips its draft generator while the controller selects k=0."""
    adapter = AdaptiveProposerAdapter(
        controller_kwargs={
            "initial_alpha": 0.60,
            "fallback_threshold": 0.65,
        }
    )

    request_id = "fallback-regression"
    draft_calls = []

    def draft_generator(k):
        draft_calls.append(k)
        return [101] * k

    for _ in range(20):
        result = adapter.propose(
            request_id=request_id,
            draft_generator=draft_generator,
            free_vram_mb=16000.0,
        )
        assert result.budget_k == 0
        assert result.skipped_draft is True
        assert result.draft_tokens == []

    assert draft_calls == []
    assert adapter.get_controller(request_id).get_rolling_alpha() == (
        pytest.approx(0.60)
    )
    assert adapter.active_request_count == 1

    # Release the session and verify cleanup.
    telemetry = adapter.end_request(request_id)

    assert len(telemetry["tree_budgets"]) == 20
    assert adapter.active_request_count == 0


def test_real_verification_feedback_can_raise_alpha():
    """Demonstrate EMA math; this is NOT periodic probing."""
    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
    )

    assert controller.select_budget().k == 0

    # Inject one simulated verification observation explicitly.
    new_alpha = controller.update(
        num_accepted=2,
        num_drafted=2,
    )

    assert new_alpha == pytest.approx(0.68)
    assert new_alpha > controller.fallback_threshold


def test_alpha_recovery_does_not_override_vram_guard():
    """Even with recovered EMA, memory safety still forces k=0."""
    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
        min_free_vram_mb=1500.0,
    )

    controller.update(num_accepted=2, num_drafted=2)

    decision = controller.select_budget(free_vram_mb=1000.0)

    assert controller.get_rolling_alpha() == pytest.approx(0.68)
    assert decision.k == 0
    assert "VRAM" in decision.reason


def test_alpha_recovery_can_resume_draft_when_cost_is_favorable():
    """Recovered EMA permits k>0 when the provided cost model favors it."""
    favorable_costs = CostProfile(
        matrix={
            (0, 1): 20.0,
            (2, 1): 21.0,
            (3, 1): 40.0,
            (4, 1): 55.0,
        }
    )

    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
        cost_profile=favorable_costs,
    )

    assert controller.select_budget().k == 0

    # Explicit simulated observation, not a probe scheduled by the code.
    controller.update(num_accepted=2, num_drafted=2)

    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=16000.0,
    )

    assert controller.get_rolling_alpha() == pytest.approx(0.68)
    assert decision.k == 2


def test_alpha_recovery_does_not_force_unprofitable_drafting():
    """Cost-model argmax may correctly retain vanilla despite EMA recovery."""
    unfavorable_costs = CostProfile(
        matrix={
            (0, 1): 20.0,
            (2, 1): 100.0,
            (3, 1): 120.0,
            (4, 1): 150.0,
        }
    )

    controller = BudgetController(
        initial_alpha=0.60,
        ema_beta=0.8,
        fallback_threshold=0.65,
        cost_profile=unfavorable_costs,
    )

    controller.update(num_accepted=2, num_drafted=2)
    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=16000.0,
    )

    assert controller.get_rolling_alpha() == pytest.approx(0.68)
    assert decision.k == 0
    assert "efficiency" in decision.reason.lower()
