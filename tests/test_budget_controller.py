"""Unit tests for BudgetController v0, CostProfile, and BudgetDecision."""
import pytest
from src.proposer.controller import CostProfile, BudgetDecision

def test_cost_profile_lookup():
    matrix = {
        (0, 1): 20.0,
        (2, 1): 25.0,
        (3, 1): 28.0,
        (4, 1): 32.0,
    }
    profile = CostProfile(matrix=matrix)
    assert profile.get_cost(k=3, batch_size=1) == 28.0
    assert profile.get_cost(k=0, batch_size=1) == 20.0

def test_cost_profile_fallback_missing_batch():
    matrix = {(0, 1): 20.0, (3, 1): 28.0}
    profile = CostProfile(matrix=matrix)
    # If batch size is not in profile, fallback to key with same k
    assert profile.get_cost(k=3, batch_size=4) == 28.0

def test_cost_profile_default_estimation():
    # Empty profile falls back to linear heuristic
    profile = CostProfile()
    assert profile.get_cost(k=0, batch_size=1) == 20.0
    assert profile.get_cost(k=3, batch_size=1) == 20.0 + 3 * 3.0


def test_estimate_expected_tokens():
    from src.proposer.controller import BudgetController

    controller = BudgetController()
    # Khi k = 0, chỉ có 1 token của vanilla
    assert controller.estimate_expected_tokens(k=0, alpha=0.9) == 1.0

    # Khi k = 1, E[tau] = 1 + alpha
    assert pytest.approx(controller.estimate_expected_tokens(k=1, alpha=0.8), 0.001) == 1.8

    # Khi k = 3, alpha = 0.5: E[tau] = 1 + 0.5 + 0.25 + 0.125 = 1.875
    assert pytest.approx(controller.estimate_expected_tokens(k=3, alpha=0.5), 0.001) == 1.875

    # Khi k = 3, alpha = 0.9: E[tau] = 1 + 0.9 + 0.81 + 0.729 = 3.439
    assert pytest.approx(controller.estimate_expected_tokens(k=3, alpha=0.9), 0.001) == 3.439


def test_rolling_alpha_updates():
    from src.proposer.controller import BudgetController

    controller = BudgetController(window_size=5, ema_beta=0.8, initial_alpha=0.8)
    assert controller.get_rolling_alpha() == 0.8

    # Bước 1: trúng 3/3 (alpha_curr = 1.0)
    # alpha_ema = 0.8 * 0.8 + 0.2 * 1.0 = 0.64 + 0.2 = 0.84
    new_alpha = controller.update(num_accepted=3, num_drafted=3)
    assert pytest.approx(new_alpha, 0.001) == 0.84
    assert pytest.approx(controller.get_rolling_alpha(), 0.001) == 0.84

    # Bước 2: trúng 0/3 (alpha_curr = 0.0)
    # alpha_ema = 0.8 * 0.84 + 0.2 * 0.0 = 0.672
    new_alpha = controller.update(num_accepted=0, num_drafted=3)
    assert pytest.approx(new_alpha, 0.001) == 0.672
    assert pytest.approx(controller.get_rolling_alpha(), 0.001) == 0.672


def test_rolling_alpha_sliding_window_bound():
    from src.proposer.controller import BudgetController

    controller = BudgetController(window_size=3)
    for _ in range(10):
        controller.update(num_accepted=2, num_drafted=3)

    assert len(controller._window) == 3


def test_select_budget_vram_pressure():
    from src.proposer.controller import BudgetController

    controller = BudgetController(min_free_vram_mb=2000.0)
    # VRAM chỉ còn 1000MB -> Phải ép k = 0
    decision = controller.select_budget(batch_size=1, free_vram_mb=1000.0)
    assert decision.k == 0
    assert "vram" in decision.reason.lower()


def test_select_budget_fallback_low_alpha():
    from src.proposer.controller import BudgetController

    controller = BudgetController(initial_alpha=0.50, fallback_threshold=0.65)
    decision = controller.select_budget(batch_size=1)
    # alpha = 0.50 < 0.65 -> ngắt về 0 (Vanilla)
    assert decision.k == 0
    assert "fallback" in decision.reason.lower() or "alpha" in decision.reason.lower()


def test_select_budget_heuristic_high_alpha():
    from src.proposer.controller import BudgetController

    controller = BudgetController(initial_alpha=0.88)
    decision = controller.select_budget(batch_size=1)
    assert decision.k in [3, 4]
    assert decision.k > 0


def test_select_budget_heuristic_mid_alpha():
    from src.proposer.controller import BudgetController

    controller = BudgetController(initial_alpha=0.72, high_threshold=0.80, fallback_threshold=0.65)
    decision = controller.select_budget(batch_size=1)
    # alpha in [0.65, 0.80) -> chọn k=2 hoặc 3 (không chọn k=4)
    assert decision.k in [2, 3]
    assert decision.k != 4


def test_select_budget_cost_model_argmax():
    from src.proposer.controller import BudgetController, CostProfile

    cost_matrix = {
        (0, 1): 20.0,
        (2, 1): 24.0,
        (3, 1): 28.0,
        (4, 1): 38.0,
    }
    profile = CostProfile(matrix=cost_matrix)
    controller = BudgetController(cost_profile=profile, initial_alpha=0.90)
    decision = controller.select_budget(batch_size=1)
    assert decision.k > 0
    assert decision.expected_tokens > 1.0


