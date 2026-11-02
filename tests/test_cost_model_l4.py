"""Unit tests for HardwareCostModel and L4 Hardware Cost Profile.

Task: [M3-02] Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4
"""
from pathlib import Path
import pytest

from src.proposer.cost_model import HardwareCostModel
from src.proposer.controller import BudgetController, CostProfile


@pytest.fixture
def l4_profile_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    profile_path = repo_root / "experiments" / "results" / "hardware_profile_l4.json"
    if not profile_path.exists():
        pytest.skip("hardware_profile_l4.json not found, run scripts/generate_l4_hardware_profile.py first")
    return profile_path


def test_hardware_cost_model_loading(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)
    assert model.metadata["device"] == "NVIDIA L4"
    assert model.metadata["sm_architecture"] == "sm_89"
    assert (0, 1) in model.matrix
    assert (3, 1) in model.matrix
    assert 1 in model.forward_target
    assert 1 in model.forward_draft


def test_forward_and_verification_costs(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)
    c_t, c_d = model.get_forward_costs(batch_size=1)
    # L4 target is approx 19.1ms, draft is approx 4.88ms
    assert pytest.approx(c_t, 0.5) == 19.1
    assert pytest.approx(c_d, 0.5) == 4.9
    assert (c_d / c_t) < 0.30

    # Verification cost C_V(k) increases slightly with k
    c_v1 = model.get_verification_cost(k=1, batch_size=1)
    c_v4 = model.get_verification_cost(k=4, batch_size=1)
    assert c_v4 > c_v1
    assert pytest.approx(c_v1, 0.1) == c_t


def test_cost_lookup_and_fallbacks(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)

    # Exact lookup
    cost_k3_b1 = model.get_cost(k=3, batch_size=1)
    assert cost_k3_b1 > 30.0  # ~34.0ms

    # Missing batch fallback: batch_size=3 should fall back to closest (b=2 or b=4)
    cost_k3_b3 = model.get_cost(k=3, batch_size=3)
    assert cost_k3_b3 > 0.0

    # Empty model generic fallback
    empty_model = HardwareCostModel()
    assert empty_model.get_cost(k=0, batch_size=1) == 20.0
    assert empty_model.get_cost(k=3, batch_size=1) == 20.0 + 3 * 5.0


def test_expected_tokens_calculation():
    assert HardwareCostModel.estimate_expected_tokens(k=0, alpha=0.9) == 1.0
    assert pytest.approx(HardwareCostModel.estimate_expected_tokens(k=1, alpha=0.8), 0.001) == 1.8
    assert pytest.approx(HardwareCostModel.estimate_expected_tokens(k=3, alpha=0.5), 0.001) == 1.875
    # Negative alpha clamped to 0
    assert HardwareCostModel.estimate_expected_tokens(k=3, alpha=-0.5) == 1.0


def test_speedup_calculation(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)

    # k=0 always has Speedup = 1.0
    assert model.calculate_speedup(k=0, alpha=0.9, batch_size=1) == 1.0

    # At low alpha (0.1), speculative decoding suffers negative speedup (< 1.0)
    low_speedup = model.calculate_speedup(k=3, alpha=0.1, batch_size=1)
    assert low_speedup < 1.0

    # At high alpha (0.95), speculative decoding achieves positive speedup (> 1.5)
    high_speedup = model.calculate_speedup(k=3, alpha=0.95, batch_size=1)
    assert high_speedup > 1.5


def test_break_even_alpha_star(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)

    # For k=2, break-even alpha* on L4 is ~0.377
    alpha_star_k2 = model.calculate_alpha_star(k=2, batch_size=1)
    assert alpha_star_k2 is not None
    assert pytest.approx(alpha_star_k2, 0.01) == 0.38

    # When alpha == alpha_star, Speedup must be approx 1.0
    speedup_at_break_even = model.calculate_speedup(k=2, alpha=alpha_star_k2, batch_size=1)
    assert pytest.approx(speedup_at_break_even, 0.02) == 1.0

    # For k=3, break-even alpha* is ~0.466
    alpha_star_k3 = model.calculate_alpha_star(k=3, batch_size=1)
    assert alpha_star_k3 is not None
    assert pytest.approx(alpha_star_k3, 0.01) == 0.47


def test_optimal_budget_selection(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)

    # Low alpha (0.20) -> should choose k=0 to prevent negative speedup
    k_opt_low, speedup_low = model.get_optimal_budget(alpha=0.20, batch_size=1)
    assert k_opt_low == 0
    assert speedup_low == 1.0

    # Moderate alpha (0.60) -> should choose speculative budget (e.g. k=2 or k=3)
    k_opt_mid, speedup_mid = model.get_optimal_budget(alpha=0.60, batch_size=1)
    assert k_opt_mid in [2, 3]
    assert speedup_mid > 1.0

    # High alpha (0.95) -> should choose higher budget (k=4) with high speedup
    k_opt_high, speedup_high = model.get_optimal_budget(alpha=0.95, batch_size=1)
    assert k_opt_high == 4
    assert speedup_high > 1.5


def test_controller_integration(l4_profile_path: Path):
    model = HardwareCostModel.from_json(l4_profile_path)
    cost_profile = model.to_cost_profile()
    assert isinstance(cost_profile, CostProfile)

    # Integrate directly into BudgetController
    controller = BudgetController(cost_profile=cost_profile, initial_alpha=0.90)
    decision = controller.select_budget(batch_size=1)
    assert decision.k in [2, 3, 4]
    assert decision.expected_tokens > 1.0
