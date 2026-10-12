"""Unit tests for the M2-02 Memory Pressure Rule (KV-cache + VRAM guard with hysteresis)."""
import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController


def make_controller(**kwargs) -> AdaptiveDraftController:
    # initial_alpha=0.80 -> Budget Rule selects k=3 when there is no memory pressure
    params = {"initial_alpha": 0.80}
    params.update(kwargs)
    return AdaptiveDraftController(**params)


def step(controller: AdaptiveDraftController, kv: float, free_vram_mb: float = 16000.0) -> int:
    return controller.select_budget(free_vram_mb=free_vram_mb, kv_cache_usage_pct=kv).k


def test_kv_cache_above_high_threshold_forces_k0_even_with_high_alpha():
    controller = make_controller(initial_alpha=0.95)

    decision = controller.select_budget(kv_cache_usage_pct=90.5)

    assert decision.k == 0
    assert "kv-cache" in decision.reason.lower()


def test_kv_cache_at_exact_high_threshold_does_not_trigger():
    controller = make_controller()

    assert step(controller, kv=90.0) == 3


def test_cooldown_holds_k0_for_minimum_steps_then_releases():
    controller = make_controller(memory_cooldown_steps=3)

    ks = [step(controller, kv) for kv in [95.0, 50.0, 50.0, 50.0]]

    # Trigger step + 2 cooldown steps = 3 steps at k=0, then Budget Rule resumes
    assert ks == [0, 0, 0, 3]


def test_no_ping_pong_around_high_threshold():
    controller = make_controller()

    ks = [step(controller, kv) for kv in [90.5, 89.9, 90.5, 89.9, 90.5, 89.9]]

    assert ks == [0, 0, 0, 0, 0, 0]


def test_hysteresis_requires_kv_below_recover_threshold_after_cooldown():
    controller = make_controller(memory_cooldown_steps=3)

    ks = [step(controller, kv) for kv in [95.0, 85.0, 85.0, 85.0, 80.0, 79.9]]

    # Cooldown elapsed at step 4 but KV stays in the [80, 90] band -> keep k=0
    assert ks == [0, 0, 0, 0, 0, 3]


def test_retrigger_during_cooldown_restarts_cooldown():
    controller = make_controller(memory_cooldown_steps=3)

    ks = [step(controller, kv) for kv in [95.0, 50.0, 95.0, 50.0, 50.0, 50.0]]

    assert ks == [0, 0, 0, 0, 0, 3]


def test_vram_guard_triggers_and_uses_same_cooldown():
    controller = make_controller(min_free_vram_mb=1500.0, memory_cooldown_steps=3)

    ks = [
        step(controller, kv=10.0, free_vram_mb=vram)
        for vram in [1000.0, 16000.0, 16000.0, 16000.0]
    ]

    assert ks == [0, 0, 0, 3]


def test_vram_recovery_requires_free_vram_above_minimum():
    controller = make_controller(min_free_vram_mb=1500.0, memory_cooldown_steps=1)

    ks = [
        step(controller, kv=10.0, free_vram_mb=vram)
        for vram in [1000.0, 1200.0, 1600.0]
    ]

    assert ks == [0, 0, 3]


def test_vram_guard_can_be_disabled():
    controller = make_controller(enable_vram_guard=False)

    assert step(controller, kv=10.0, free_vram_mb=500.0) == 3


def test_reset_clears_memory_pressure_state():
    controller = make_controller()
    step(controller, kv=95.0)

    controller.reset()

    assert step(controller, kv=10.0) == 3
    assert controller.get_stats()["memory_pressure_active"] is False


def test_stats_expose_memory_pressure_state():
    controller = make_controller()

    step(controller, kv=95.0)

    assert controller.get_stats()["memory_pressure_active"] is True


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        make_controller(kv_cache_high_pct=80.0, kv_cache_recover_pct=90.0)
    with pytest.raises(ValueError):
        make_controller(memory_cooldown_steps=0)


def test_adapter_passes_kv_cache_usage_and_skips_draft():
    adapter = AdaptiveProposerAdapter(controller_kwargs={"initial_alpha": 0.95})
    draft_calls = []

    def draft_generator(k):
        draft_calls.append(k)
        return [7] * k

    result = adapter.propose(
        request_id="m2-02-kv-pressure",
        draft_generator=draft_generator,
        kv_cache_usage_pct=92.0,
    )

    assert result.budget_k == 0
    assert result.skipped_draft is True
    assert draft_calls == []
