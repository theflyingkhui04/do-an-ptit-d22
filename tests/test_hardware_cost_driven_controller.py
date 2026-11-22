"""Tests for hardware cost-driven AdaptiveDraftController and non-monotonicity guards (R6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def test_inspect_monotonicity_detects_a10g_small_batch_inversions():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    anomalies = inspect_cost_matrix_monotonicity(data["cost_matrix_ms"])
    below_k0_pairs = {(item["batch"], item["k"]) for item in anomalies if item["below_k0"]}
    assert (1, 1) in below_k0_pairs
    assert (2, 1) in below_k0_pairs
    assert (2, 2) in below_k0_pairs
    assert (4, 1) in below_k0_pairs
    assert (4, 2) in below_k0_pairs
    assert all(item["batch"] not in {8, 16, 32} for item in anomalies if item["below_k0"])


def test_inspect_monotonicity_returns_empty_for_strictly_increasing_matrix():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    clean = {
        "k0_b1": 15.0,
        "k1_b1": 16.0,
        "k2_b1": 18.0,
        "k0_b8": 16.0,
        "k1_b8": 20.0,
        "k2_b8": 25.0,
    }
    assert inspect_cost_matrix_monotonicity(clean) == []
