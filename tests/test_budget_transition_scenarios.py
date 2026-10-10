
"""M2-01: Dynamic tree-budget transition tests.

Target commit:
ddf24d194e13b08b5033109d192184f7f55518d5

Scope:
- High acceptance
- Medium acceptance
- Low acceptance / vanilla fallback
- Per-request budget history
- VRAM safety override

These are CPU-level simulation tests, not GPU benchmarks.
"""

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import BudgetController, CostProfile


def make_profile():
    """Deterministic synthetic costs in milliseconds."""
    return CostProfile(
        matrix={
            (0, 1): 20.0,
            (2, 1): 21.0,
            (3, 1): 24.0,
            (4, 1): 27.0,
        }
    )


@pytest.mark.parametrize(
    "alpha, expected_k",
    [
        (0.90, 4),
        (0.72, 3),
        (0.50, 0),
    ],
)
def test_budget_selection_at_different_alpha(
    alpha, expected_k
):
    controller = BudgetController(
        initial_alpha=alpha,
        cost_profile=make_profile(),
        fallback_threshold=0.65,
        high_threshold=0.80,
    )

    decision = controller.select_budget(
        batch_size=1,
        free_vram_mb=16000.0,
    )

    assert decision.k == expected_k
    assert decision.alpha_ema == pytest.approx(alpha)


def test_tree_budget_changes_with_acceptance():
    """Simulate high -> medium -> low alpha in one request."""

    adapter = AdaptiveProposerAdapter(
        controller_kwargs={
            "initial_alpha": 0.90,
            "ema_beta": 0.80,
            "fallback_threshold": 0.65,
            "high_threshold": 0.80,
            "cost_profile": make_profile(),
        }
    )

    request_id = "m2-01-alpha-transition"
    generated_budgets = []

    def draft_generator(k):
        generated_budgets.append(k)
        return [101] * k

    # Stage 1: High acceptance.
    first = adapter.propose(
        request_id=request_id,
        draft_generator=draft_generator,
        free_vram_mb=16000.0,
    )

    assert first.budget_k == 4
    assert first.skipped_draft is False

    adapter.on_verification_completed(
        request_id,
        num_accepted=4,
        num_drafted=4,
    )

    # EMA = 0.8 * 0.90 + 0.2 * 1.0 = 0.92
    controller = adapter.get_controller(request_id)
    assert controller.get_rolling_alpha() == pytest.approx(0.92)

    # Stage 2: Still high acceptance.
    second = adapter.propose(
        request_id=request_id,
        draft_generator=draft_generator,
        free_vram_mb=16000.0,
    )

    assert second.budget_k == 4

    # Verification rejects every drafted token.
    adapter.on_verification_completed(
        request_id,
        num_accepted=0,
        num_drafted=second.budget_k,
    )

    # EMA = 0.8 * 0.92 = 0.736
    assert controller.get_rolling_alpha() == pytest.approx(0.736)

    # Stage 3: Medium acceptance.
    third = adapter.propose(
        request_id=request_id,
        draft_generator=draft_generator,
        free_vram_mb=16000.0,
    )

    assert third.budget_k == 3
    assert third.skipped_draft is False

    adapter.on_verification_completed(
        request_id,
        num_accepted=0,
        num_drafted=third.budget_k,
    )

    # EMA = 0.8 * 0.736 = 0.5888
    assert controller.get_rolling_alpha() == pytest.approx(0.5888)

    # Stage 4: Low acceptance -> vanilla fallback.
    fourth = adapter.propose(
        request_id=request_id,
        draft_generator=draft_generator,
        free_vram_mb=16000.0,
    )

    assert fourth.budget_k == 0
    assert fourth.skipped_draft is True
    assert fourth.draft_tokens == []

    # No draft should be generated during vanilla fallback.
    assert generated_budgets == [4, 4, 3]

    # Verify complete per-request telemetry.
    stats = controller.get_stats()

    assert stats["tree_budgets"] == [4, 4, 3, 0]
    assert stats["step_count"] == 3
    assert stats["alpha_ema"] == pytest.approx(0.5888)
    assert stats["mean_k"] == pytest.approx(2.75)

    telemetry = adapter.end_request(request_id)

    assert telemetry["request_id"] == request_id
    assert telemetry["lane"] == "adaptive"
    assert telemetry["tree_budgets"] == [4, 4, 3, 0]
    assert telemetry["alpha_ema"] == pytest.approx(0.5888)
    assert adapter.active_request_count == 0


def test_vram_pressure_overrides_high_alpha():
    """High acceptance must not override memory safety."""

    adapter = AdaptiveProposerAdapter(
        controller_kwargs={
            "initial_alpha": 0.95,
            "cost_profile": make_profile(),
            "min_free_vram_mb": 1500.0,
        }
    )

    request_id = "m2-01-vram-override"
    draft_calls = []

    def draft_generator(k):
        draft_calls.append(k)
        return [101] * k

    result = adapter.propose(
        request_id=request_id,
        draft_generator=draft_generator,
        free_vram_mb=1000.0,
    )

    assert result.budget_k == 0
    assert result.skipped_draft is True
    assert result.draft_tokens == []
    assert draft_calls == []

    telemetry = adapter.end_request(request_id)

    assert telemetry["tree_budgets"] == [0]
    assert adapter.active_request_count == 0
