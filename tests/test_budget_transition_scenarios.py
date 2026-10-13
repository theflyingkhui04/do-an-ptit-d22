
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
        controller_cls=BudgetController,
        controller_kwargs={
            "initial_alpha": 0.90,
            "ema_beta": 0.80,
            "fallback_threshold": 0.65,
            "high_threshold": 0.80,
            "cost_profile": make_profile(),
        },
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


def test_smoke_ten_requests_state_transitions():
    """Verify state transitions across 10 requests (5 W1 + 5 W3) with probe_interval=20.

    Checklist 5: Ensure controller smoothly transitions among k in {0, 2, 3, 4}
    without crashes, and verifies telemetry generation.
    """
    from src.server.budget_planner import AdaptiveBudgetPlanner

    adapter = AdaptiveProposerAdapter()
    planner = AdaptiveBudgetPlanner(adapter=adapter)

    # 10 synthetic requests: 5 W1 (high acceptance) and 5 W3 (low acceptance)
    all_k_observed = set()

    # Pass 1: 5 requests with W1 profile (high alpha -> k transitions to 3, 4)
    for i in range(5):
        req_id = f"smoke_w1_{i:03d}"
        for step in range(25):
            plan = planner.plan_step(decode_req_ids=[req_id], batch_size=1, kv_usage=0.20, free_vram_mb=4000.0)
            k = plan.per_request_k[req_id]
            all_k_observed.add(k)
            # High acceptance verification: accept all k tokens
            planner.record_verification(req_id, num_drafted=k, num_accepted=k)
        telemetry = planner.finish_request(req_id)
        assert telemetry is not None
        assert telemetry["mean_accept_len"] >= 2.0
        assert set(telemetry["tree_budgets"]).issubset({2, 3, 4})

    # Pass 2: 5 requests with W3 profile (low alpha -> fallback to k=0, probes k=2 at step 20)
    for i in range(5):
        req_id = f"smoke_w3_{i:03d}"
        for step in range(25):
            plan = planner.plan_step(decode_req_ids=[req_id], batch_size=1, kv_usage=0.20, free_vram_mb=4000.0)
            k = plan.per_request_k[req_id]
            all_k_observed.add(k)
            # Low acceptance verification: 0 tokens accepted
            planner.record_verification(req_id, num_drafted=k, num_accepted=0)
        telemetry = planner.finish_request(req_id)
        assert telemetry is not None
        # With probe_interval=20, step 20 should trigger periodic probe k=2
        assert 0 in telemetry["tree_budgets"]
        assert 2 in telemetry["tree_budgets"]

    # All candidate k in {0, 2, 3, 4} must be observed across the 10 requests
    assert all_k_observed == {0, 2, 3, 4}

