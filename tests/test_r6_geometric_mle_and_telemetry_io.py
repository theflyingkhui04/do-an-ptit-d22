"""Tests for R6 Discounted-Count Geometric MLE alpha estimator and batched step telemetry I/O."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _load_a10g_profile():
    from src.proposer.controller import EmpiricalCostProfile

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


@pytest.mark.parametrize("true_p", [0.40, 0.60, 0.80])
@pytest.mark.parametrize("k_fixed", [1, 2, 3, 4])
def test_geometric_mle_is_unbiased_across_all_k(true_p, k_fixed):
    """AC-01 & NFR-01: get_geometric_alpha() hội tụ về p thật (±0.02) ở mọi K in {1..4}."""
    from src.proposer.controller import AdaptiveDraftController

    rng = random.Random(2026 + k_fixed * 100 + int(true_p * 100))
    ctrl = AdaptiveDraftController(
        initial_alpha=0.8,
        mle_beta=0.90,
        mle_prior_weight=2.0,
        alpha_estimator="geometric_mle",
    )
    estimates = []
    for step in range(4000):
        acc = 0
        for _ in range(k_fixed):
            if rng.random() < true_p:
                acc += 1
            else:
                break
        ctrl.update(num_accepted=acc, num_drafted=k_fixed)
        if step >= 500:
            estimates.append(ctrl.get_geometric_alpha())
    mean_est = sum(estimates) / len(estimates)
    assert abs(mean_est - true_p) <= 0.02


def test_geometric_mle_avoids_step1_deadlock_but_falls_back_on_hard_prompt():
    """AC-02: Bước 1 trượt không gây khóa K=0; prompt khó liên tiếp trượt vẫn về K=0."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.80,
        candidate_budgets=budgets,
        cost_profile=profile,
        mle_beta=0.90,
        mle_prior_weight=2.0,
    )
    ctrl.update(num_accepted=0, num_drafted=4)
    assert ctrl.get_geometric_alpha() > 0.50
    d1 = ctrl.select_budget(batch_size=1)
    assert d1.k > 0
    assert d1.alpha_estimator == "geometric_mle"

    # Prompt khó (luôn trượt a=0): B=16 rơi về K=0 trong <= 3 bước; B=1 trong <= 15 bước
    ctrl_b16 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b16_k0_step = None
    for s in range(1, 6):
        ctrl_b16.update(0, 2)
        if ctrl_b16.select_budget(batch_size=16).k == 0:
            b16_k0_step = s
            break
    assert b16_k0_step is not None and b16_k0_step <= 3

    ctrl_b1 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b1_k0_step = None
    for s in range(1, 20):
        ctrl_b1.update(0, 1)
        if ctrl_b1.select_budget(batch_size=1).k == 0:
            b1_k0_step = s
            break
    assert b1_k0_step is not None and b1_k0_step <= 15


def test_geometric_mle_keeps_k3_k4_dominant_at_low_batch_and_k0_at_high_batch():
    """AC-03: Mô phỏng 200 request 60 token trên A10G thật (p=0.60)."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    for batch_size in (1, 8, 32):
        rng = random.Random(123 + batch_size)
        counts = Counter()
        for _ in range(200):
            ctrl = AdaptiveDraftController(
                initial_alpha=0.80,
                candidate_budgets=budgets,
                cost_profile=profile,
            )
            gen = 0
            while gen < 60:
                k = ctrl.select_budget(batch_size=batch_size).k
                counts[k] += 1
                acc = 0
                for _ in range(k):
                    if rng.random() < 0.60:
                        acc += 1
                    else:
                        break
                gen += acc + 1
                if k > 0:
                    ctrl.update(acc, k)
        tot = sum(counts.values())
        if batch_size == 1:
            assert (counts[3] + counts[4]) / tot >= 0.80
            assert counts[0] / tot < 0.02
        elif batch_size == 8:
            assert counts[3] / tot >= 0.75
        else:
            assert counts[0] / tot >= 0.90


def test_mle_parameter_validation_clamping_and_reset():
    """Review Focus #1 & #2: Validate tham số bool/out-of-range, kẹp input xấu và reset sạch."""
    from src.proposer.controller import AdaptiveDraftController

    for bad_kwargs in (
        {"mle_beta": True},
        {"mle_beta": 0.0},
        {"mle_beta": 1.0},
        {"mle_beta": float("nan")},
        {"mle_prior_weight": True},
        {"mle_prior_weight": -0.5},
        {"alpha_estimator": "unknown"},
    ):
        with pytest.raises(ValueError):
            AdaptiveDraftController(**bad_kwargs)

    ctrl = AdaptiveDraftController(initial_alpha=0.75, mle_prior_weight=2.0, alpha_estimator="geometric_mle")
    ctrl.update(num_accepted=-5, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    ctrl.update(num_accepted=10, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    before_zero = ctrl.get_geometric_alpha()
    ctrl.update(num_accepted=0, num_drafted=0)
    assert ctrl.get_geometric_alpha() == pytest.approx(before_zero)

    ctrl.reset(initial_alpha=0.65)
    assert ctrl.get_geometric_alpha() == pytest.approx(0.65)
    assert ctrl.get_stats()["total_tested_positions"] == 0
