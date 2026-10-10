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
