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


def test_controller_reset_and_stats():
    from src.proposer.controller import BudgetController

    controller = BudgetController(initial_alpha=0.8)
    controller.select_budget()
    controller.update(num_accepted=3, num_drafted=3)
    controller.select_budget()

    stats = controller.get_stats()
    assert "tree_budgets" in stats
    assert len(stats["tree_budgets"]) == 2
    assert stats["step_count"] == 1
    assert "mean_k" in stats
    assert "alpha_ema" in stats

    controller.reset(initial_alpha=0.75)
    assert controller.get_rolling_alpha() == 0.75
    assert controller.get_stats()["tree_budgets"] == []
    assert controller.get_stats()["step_count"] == 0


def test_adaptive_draft_controller_inheritance_and_defaults():
    from src.proposer.controller import AdaptiveDraftController, BudgetController

    controller = AdaptiveDraftController()
    assert isinstance(controller, BudgetController)
    assert controller.window_size == 10
    assert controller.ema_beta == 0.8
    assert controller.initial_alpha == 0.8
    assert controller.get_window_samples() == []
    assert controller.step_count == 0


def test_adaptive_draft_controller_window_tracking():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(window_size=10)
    for i in range(15):
        controller.update(num_accepted=i % 4, num_drafted=3)

    samples = controller.get_window_samples()
    assert len(samples) == 10
    assert controller.step_count == 15
    assert len(controller._window) == 10


def test_adaptive_draft_controller_unbiased_warmup():
    from src.proposer.controller import AdaptiveDraftController

    # Test trường hợp prompt khó: bước 1 trượt sạch (0/3 accepted)
    controller = AdaptiveDraftController(window_size=10, ema_beta=0.8, initial_alpha=0.8)
    # Chưa update bước nào -> effective_alpha = initial_alpha
    assert controller.get_effective_alpha() == 0.8

    # Bước 1: 0/3 -> step_alpha = 0.0
    controller.update(num_accepted=0, num_drafted=3)
    # EMA thông thường bị ghìm ở 0.8 * 0.8 + 0.2 * 0 = 0.64
    # Nhưng unbiased alpha phải loại bỏ hoàn toàn initial_alpha và bằng chính xác 0.0
    assert pytest.approx(controller.get_effective_alpha(), 0.01) == 0.0

    # Bước 2: 3/3 -> step_alpha = 1.0
    controller.update(num_accepted=3, num_drafted=3)
    # Với beta = 0.8, mẫu [0.0, 1.0], unbiased alpha = (0.8*0 + 0.2*1.0) / (1 - 0.8^2) = 0.2 / 0.36 = 0.555
    assert pytest.approx(controller.get_effective_alpha(), 0.01) == 0.555


def test_adaptive_draft_controller_window_stats():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(window_size=5, initial_alpha=0.8)
    stats_empty = controller.get_window_stats()
    assert stats_empty["is_warmed_up"] is False
    assert stats_empty["mean"] == 0.8
    assert stats_empty["variance"] == 0.0
    assert stats_empty["samples"] == []

    # Bơm 3 mẫu: 0.6, 0.8, 1.0
    controller.update(num_accepted=3, num_drafted=5)  # 0.6
    controller.update(num_accepted=4, num_drafted=5)  # 0.8
    controller.update(num_accepted=5, num_drafted=5)  # 1.0

    stats = controller.get_window_stats()
    assert stats["is_warmed_up"] is False  # 3 < 5
    assert pytest.approx(stats["mean"], 0.001) == 0.8
    assert pytest.approx(stats["min"], 0.001) == 0.6
    assert pytest.approx(stats["max"], 0.001) == 1.0
    assert stats["variance"] == pytest.approx(0.02667, abs=1e-4)
    assert len(stats["samples"]) == 3

    # Bơm thêm 2 mẫu để đầy cửa sổ 5 bước
    controller.update(num_accepted=5, num_drafted=5)
    controller.update(num_accepted=5, num_drafted=5)
    stats_full = controller.get_window_stats()
    assert stats_full["is_warmed_up"] is True
    assert len(stats_full["samples"]) == 5


def test_adaptive_draft_controller_budget_rules():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(fallback_threshold=0.65, high_threshold=0.80)

    # 1. High acceptance (>= 0.80) -> k = 3 (hoặc 4 nếu rất cao >= 0.90)
    k, reason = controller.apply_budget_rule(alpha=0.85)
    assert k == 3
    assert "high" in reason.lower()

    k_high, _ = controller.apply_budget_rule(alpha=0.92)
    assert k_high in [3, 4]

    # 2. Mid acceptance [0.65, 0.80) -> k = 2
    k_mid, reason_mid = controller.apply_budget_rule(alpha=0.72)
    assert k_mid == 2
    assert "mid" in reason_mid.lower() or "throttle" in reason_mid.lower()

    # 3. Low acceptance (< 0.65) -> k = 0
    k_low, reason_low = controller.apply_budget_rule(alpha=0.55)
    assert k_low == 0
    assert "low" in reason_low.lower() or "fallback" in reason_low.lower()


def test_adaptive_draft_controller_select_budget_flow():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController()

    # Initial state (effective_alpha = 0.80) -> k=3
    decision = controller.select_budget()
    assert decision.k == 3

    # VRAM pressure -> ép k=0
    decision_vram = controller.select_budget(free_vram_mb=1000.0)
    assert decision_vram.k == 0
    assert "vram" in decision_vram.reason.lower()

    # Gặp câu khó: step 1 trượt sạch -> k lập tức rơi về 0 (chống tụt tốc độ ngay)
    controller.update(num_accepted=0, num_drafted=3)
    decision_step1 = controller.select_budget(free_vram_mb=16000.0)
    assert decision_step1.k == 0
    assert decision_step1.alpha_ema < 0.65


def test_adaptive_draft_controller_periodic_probing_prevents_deadlock():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(
        initial_alpha=0.50,
        probe_interval=5,
        probe_k=2,
    )

    # 4 consecutive steps at low alpha (0.50) -> k = 0
    for i in range(4):
        k, reason = controller.apply_budget_rule(alpha=0.50)
        assert k == 0
        assert "fallback" in reason.lower()
        assert controller.consecutive_fallback_steps == i + 1

    # 5th step reaches probe_interval -> triggers probe with k=2
    k_probe, reason_probe = controller.apply_budget_rule(alpha=0.50)
    assert k_probe == 2
    assert "periodic probe" in reason_probe.lower()
    assert controller.consecutive_fallback_steps == 0

    # 6th step -> back to k=0 if alpha still low
    k_after, _ = controller.apply_budget_rule(alpha=0.50)
    assert k_after == 0
    assert controller.consecutive_fallback_steps == 1

    # When alpha recovers (>= 0.65), counter resets to 0
    k_rec, _ = controller.apply_budget_rule(alpha=0.75)
    assert k_rec == 2
    assert controller.consecutive_fallback_steps == 0




