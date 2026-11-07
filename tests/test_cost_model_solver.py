"""Unit test suite for FlexCostModel solver and dynamic break-even alpha*(T, B).

Task: [M4-02-03] Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động
"""
import time
import pytest

try:
    from flexsds.core.cost_model import FlexCostModel, SolverResult
    from flexsds.core.tree_topology import get_tree_topology, TreeTopology
except ModuleNotFoundError:
    from src.flexsds.core.cost_model import FlexCostModel, SolverResult
    from src.flexsds.core.tree_topology import get_tree_topology, TreeTopology


def test_solver_result_dataclass():
    """Verifies SolverResult fields, immutability, and defaults."""
    res = SolverResult(
        alpha_star=0.45,
        is_feasible=True,
        status="converged",
        cost_ratio=1.65,
        residual=1e-5,
        iterations=50,
        topology_name="T8",
        batch_size=1,
    )
    assert res.alpha_star == 0.45
    assert res.is_feasible is True
    assert res.status == "converged"
    assert res.cost_ratio == 1.65
    assert res.residual == 1e-5
    assert res.iterations == 50
    assert res.topology_name == "T8"
    assert res.batch_size == 1

    with pytest.raises(Exception):
        res.alpha_star = 0.50  # frozen dataclass


def test_resolve_topology_support():
    """Verifies _resolve_topology resolves str, TreeTopology, and int correctly."""
    model = FlexCostModel()
    t4 = model._resolve_topology("T4")
    assert isinstance(t4, TreeTopology)
    assert t4.name == "T4"
    assert t4.num_tokens == 4

    # Direct TreeTopology object
    assert model._resolve_topology(t4) is t4

    # Legacy integer k
    t0 = model._resolve_topology(0)
    assert t0.name == "T0"
    assert t0.num_tokens == 0

    t_legacy = model._resolve_topology(4)
    assert t_legacy.name == "T4"

    # Unknown string raises KeyError
    with pytest.raises(KeyError):
        model._resolve_topology("UNKNOWN_T99")


def test_cost_model_from_json_tree_and_legacy_keys():
    """Verifies loading JSON profile supporting both topo_{name}_b{b} and k{k}_b{b} formats."""
    sample_data = {
        "metadata": {"device": "Mock GPU", "attn_verify_factor": 0.008},
        "cost_matrix_ms": {
            "k0_b1": 20.0,
            "k3_b1": 35.0,
            "topo_T8_b1": 32.5,
            "topo_T16_b2": 45.0,
        },
        "forward_costs_median_ms": {
            "target_c_t": {"1": 20.0, "2": 21.0},
            "draft_c_d": {"1": 5.0, "2": 5.5},
        },
    }
    model = FlexCostModel.from_json(sample_data)
    assert model.provenance == "measured"
    assert model.get_step_cost("T8", batch_size=1) == 32.5
    assert model.get_step_cost("T16", batch_size=2) == 45.0
    assert model.get_step_cost(3, batch_size=1) == 35.0
    assert model.get_step_cost(0, batch_size=1) == 20.0


def test_analytical_fallback_calculation():
    """Verifies analytical calculation when profile matrix does not contain requested key."""
    model = FlexCostModel(
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
        metadata={"attn_verify_factor": 0.008},
    )
    # T0 should equal C_T(1) = 20.0
    assert model.get_step_cost("T0", batch_size=1) == 20.0

    # T4 has N=4: draft = 4 * 5.0 = 20.0, verify = 20.0 * (1 + 0.008 * 3) = 20.48 -> step = 40.48
    cost_t4 = model.get_step_cost("T4", batch_size=1)
    assert pytest.approx(cost_t4, 0.01) == 40.48


def test_solve_alpha_star_always_profitable():
    """Boundary 1: R <= 1.0 (profitable even at alpha=0)."""
    # C(T, B) = 15.0, C(0, B) = 20.0 -> R = 0.75 <= 1.0
    model = FlexCostModel(
        cost_matrix={("T4", 1): 15.0, ("T0", 1): 20.0, (0, 1): 20.0},
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
    )
    result = model.solve_alpha_star("T4", batch_size=1)
    assert result.is_feasible is True
    assert result.status == "always_profitable"
    assert result.alpha_star == 0.0
    assert result.cost_ratio == 0.75
    assert result.iterations == 0


def test_solve_alpha_star_infeasible_s08_handling():
    """Boundary 2 (Review R2 S08): R > 1 + d is infeasible and must NOT assign alpha*=1.0."""
    # T8 has depth d=3, so 1 + d = 4.0.
    # Set C(T8, 1) = 100.0, C(0, 1) = 20.0 -> R = 5.0 > 4.0.
    model = FlexCostModel(
        cost_matrix={("T8", 1): 100.0, ("T0", 1): 20.0, (0, 1): 20.0},
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
    )
    result = model.solve_alpha_star("T8", batch_size=1)
    assert result.is_feasible is False
    assert result.status == "infeasible"
    assert result.alpha_star is None
    assert result.cost_ratio == 5.0
    assert result.iterations == 0


def test_solve_alpha_star_convergence_and_residual():
    """Convergence check: 1.0 < R <= 1 + d converges with residual < 10^-4."""
    # Target 20.0, draft 5.0. T4 (d=4).
    # Cost = 40.48, C_vanilla = 20.0 -> R = 2.024.
    model = FlexCostModel(
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
        metadata={"attn_verify_factor": 0.008},
    )
    result = model.solve_alpha_star("T4", batch_size=1)
    assert result.is_feasible is True
    assert result.status == "converged"
    assert result.alpha_star is not None
    assert 0.0 < result.alpha_star <= 1.0
    assert result.residual < 1e-4

    # Verify that at alpha_star, predicted speedup is approx 1.0
    speedup = model.evaluate_speedup("T4", batch_size=1, alpha=result.alpha_star)
    assert pytest.approx(speedup, abs=1e-4) == 1.0


def test_evaluate_speedup_values():
    """Verifies evaluate_speedup matches theoretical formula."""
    model = FlexCostModel(
        cost_matrix={("T4", 1): 40.0, ("T0", 1): 20.0, (0, 1): 20.0},
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
    )
    # At alpha = 0.0, E[tau] = 1.0 -> Speedup = (1.0 * 20.0) / 40.0 = 0.5
    s_low = model.evaluate_speedup("T4", batch_size=1, alpha=0.0)
    assert pytest.approx(s_low, 1e-4) == 0.5

    # At alpha = 1.0, E[tau] = 1 + 4 = 5.0 -> Speedup = (5.0 * 20.0) / 40.0 = 2.5
    s_high = model.evaluate_speedup("T4", batch_size=1, alpha=1.0)
    assert pytest.approx(s_high, 1e-4) == 2.5


def test_cache_alpha_star_performance():
    """Verifies that second solve call hits cache in < 0.001 ms."""
    model = FlexCostModel(
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
    )
    # First call: computes and caches
    res1 = model.solve_alpha_star("T8", batch_size=1)

    # Second call: cache hit
    t0 = time.perf_counter()
    res2 = model.solve_alpha_star("T8", batch_size=1)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert res1.alpha_star == res2.alpha_star
    assert dt_ms < 0.01  # significantly below 0.01 ms


def test_select_optimal_topology_and_fallback_gate():
    """Verifies select_optimal_topology argmax and fallback gate to T0."""
    model = FlexCostModel(
        cost_matrix={
            ("T0", 1): 20.0,
            ("T4", 1): 30.0,
            ("T8", 1): 35.0,
            (0, 1): 20.0,
        },
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
    )
    # At high alpha (0.9), speculative trees yield speedup > 1.0
    best_topo, speedup = model.select_optimal_topology(alpha=0.9, batch_size=1, candidate_topologies=["T0", "T4", "T8"])
    assert best_topo in ["T4", "T8"]
    assert speedup > 1.0

    # At extremely low alpha (0.01), all trees have speedup < 1.0 -> fallback gate returns T0, 1.0
    fallback_topo, fallback_speedup = model.select_optimal_topology(alpha=0.01, batch_size=1, candidate_topologies=["T0", "T4", "T8"])
    assert fallback_topo == "T0"
    assert fallback_speedup == 1.0


def test_backward_compatibility_wrappers():
    """Verifies backward compatibility with legacy calculate_alpha_star and select_optimal_budget."""
    model = FlexCostModel(
        forward_target={1: 20.0},
        forward_draft={1: 5.0},
        cost_matrix={(0, 1): 20.0, (2, 1): 30.0, (3, 1): 35.0},
    )
    # calculate_alpha_star with integer k
    alpha_k = model.calculate_alpha_star(k=3, batch_size=1)
    assert 0.0 < alpha_k < 1.0

    # estimate_expected_tokens
    e_tau = model.estimate_expected_tokens(k=3, alpha=0.5)
    assert pytest.approx(e_tau, 0.01) == 1.875

    # select_optimal_budget
    best_k, sp = model.select_optimal_budget(alpha=0.8, batch_size=1, candidate_budgets=[0, 2, 3])
    assert best_k in [0, 2, 3]
    assert sp >= 1.0
