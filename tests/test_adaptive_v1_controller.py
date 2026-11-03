"""[Adaptive v1] Draft floor, warm-up guard and decision telemetry of AdaptiveDraftController.

Defaults must keep v0 behaviour; v1 behaviour is opt-in through controller kwargs.
No vLLM server or GPU is required.
"""

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController


def bad_first_sample(controller):
    controller.select_budget()
    controller.update(num_accepted=0, num_drafted=3)


# ------------------------------------------------------------- defaults stay v0
def test_default_low_alpha_still_falls_back_to_zero():
    controller = AdaptiveDraftController(initial_alpha=0.5)

    assert controller.select_budget().k == 0


def test_default_single_bad_first_sample_drops_to_zero():
    controller = AdaptiveDraftController(initial_alpha=0.8)
    bad_first_sample(controller)

    assert controller.get_effective_alpha() == pytest.approx(0.0)
    assert controller.select_budget().k == 0


# ------------------------------------------------------------- F1: draft floor
def test_floor_replaces_zero_when_alpha_is_low():
    controller = AdaptiveDraftController(initial_alpha=0.5, min_k=2)

    decisions = [controller.select_budget() for _ in range(25)]

    assert all(d.k == 2 for d in decisions)
    assert "floor" in decisions[0].reason.lower()


def test_floor_does_not_override_memory_pressure():
    controller = AdaptiveDraftController(initial_alpha=0.95, min_k=2)

    assert controller.select_budget(kv_cache_usage_pct=95.0).k == 0


def test_floor_must_be_a_candidate_budget():
    with pytest.raises(ValueError, match="min_k"):
        AdaptiveDraftController(min_k=1)


def test_floor_keeps_high_alpha_rules():
    controller = AdaptiveDraftController(initial_alpha=0.95, min_k=2)

    assert controller.select_budget().k == 4


# ------------------------------------------------------------- F2: warm-up guard
def test_warmup_guard_ignores_single_bad_first_sample():
    controller = AdaptiveDraftController(initial_alpha=0.8, min_warmup_samples=3)
    bad_first_sample(controller)

    assert controller.select_budget().k == 3


def test_warmup_guard_hands_over_after_enough_samples():
    controller = AdaptiveDraftController(initial_alpha=0.8, min_warmup_samples=3)
    for _ in range(3):
        controller.select_budget()
        controller.update(num_accepted=0, num_drafted=3)

    assert controller.select_budget().k == 0


def test_warmup_guard_rejects_invalid_value():
    with pytest.raises(ValueError, match="min_warmup_samples"):
        AdaptiveDraftController(min_warmup_samples=0)


# ------------------------------------------------------------- decision telemetry
def test_reason_counts_by_category():
    controller = AdaptiveDraftController(initial_alpha=0.95, min_k=2)
    controller.select_budget()
    controller.select_budget(kv_cache_usage_pct=95.0)
    controller.update(num_accepted=0, num_drafted=4)
    controller.update(num_accepted=0, num_drafted=4)

    counts = controller.get_stats()["reason_counts"]

    assert counts["high"] == 1
    assert counts["memory"] == 1
    assert sum(counts.values()) == 2


def test_reason_counts_cover_fallback_probe_and_floor():
    v0 = AdaptiveDraftController(initial_alpha=0.5, probe_interval=3)
    for _ in range(3):
        v0.select_budget()
    floor = AdaptiveDraftController(initial_alpha=0.5, min_k=2)
    floor.select_budget()

    assert v0.get_stats()["reason_counts"] == {"fallback": 2, "probe": 1}
    assert floor.get_stats()["reason_counts"] == {"floor": 1}


def test_adapter_telemetry_carries_reason_counts():
    adapter = AdaptiveProposerAdapter(controller_kwargs={"initial_alpha": 0.95})
    adapter.get_controller("r1").select_budget()

    assert adapter.end_request("r1")["reason_counts"] == {"high": 1}
