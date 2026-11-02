"""Hardware-aware Cost Model module for Speculative Decoding.

Task: [M3-02] Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4
Calculates exact step latency C(k, B), expected tokens E[tau], speedup ratios,
and dynamically solves break-even acceptance rates (alpha_star) to prevent negative speedup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.proposer.controller import CostProfile


@dataclass
class HardwareCostModel:
    """Hardware cost model evaluating execution latencies and break-even thresholds.
    
    Attributes:
        matrix: Lookup table mapping (k, batch_size) -> step_cost_ms.
        forward_target: Lookup table mapping batch_size -> target_step_ms (C_T).
        forward_draft: Lookup table mapping batch_size -> draft_step_ms (C_D).
        verification_costs: Lookup table mapping (k, batch_size) -> verification_cost_ms (C_V).
        break_even_alpha: Precomputed or dynamically solved alpha_star(k, B).
        metadata: Device, framework, and benchmark provenance details.
    """
    matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    forward_target: Dict[int, float] = field(default_factory=dict)
    forward_draft: Dict[int, float] = field(default_factory=dict)
    verification_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    break_even_alpha: Dict[Tuple[int, int], Optional[float]] = field(default_factory=dict)
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path_or_dict: Union[str, Path, dict]) -> HardwareCostModel:
        """Loads and parses a hardware profile from a JSON file path or dictionary."""
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        matrix: Dict[Tuple[int, int], float] = {}
        for key_str, val in data.get("cost_matrix_ms", {}).items():
            # Key format: 'k<k>_b<b>'
            parts = key_str.split("_")
            if len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
                matrix[(k_val, b_val)] = float(val)

        v_costs: Dict[Tuple[int, int], float] = {}
        for key_str, val in data.get("verification_costs_ms", {}).items():
            parts = key_str.split("_")
            if len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
                v_costs[(k_val, b_val)] = float(val)

        c_target: Dict[int, float] = {}
        f_costs = data.get("forward_costs_median_ms", {})
        for b_str, val in f_costs.get("target_c_t", {}).items():
            c_target[int(b_str)] = float(val)

        c_draft: Dict[int, float] = {}
        for b_str, val in f_costs.get("draft_c_d", {}).items():
            c_draft[int(b_str)] = float(val)

        alpha_stars: Dict[Tuple[int, int], Optional[float]] = {}
        for key_str, val in data.get("break_even_alpha", {}).items():
            parts = key_str.split("_")
            if len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
                alpha_stars[(k_val, b_val)] = float(val) if val is not None else None

        return cls(
            matrix=matrix,
            forward_target=c_target,
            forward_draft=c_draft,
            verification_costs=v_costs,
            break_even_alpha=alpha_stars,
            metadata=data.get("metadata", {}),
        )

    def get_cost(self, k: int, batch_size: int = 1) -> float:
        """Gets step execution latency C(k, B) in milliseconds with robust fallbacks."""
        if (k, batch_size) in self.matrix:
            return self.matrix[(k, batch_size)]

        # Fallback 1: Closest batch size with same k
        matching_k = [(b, cost) for (cand_k, b), cost in self.matrix.items() if cand_k == k]
        if matching_k:
            closest_b, closest_cost = min(matching_k, key=lambda item: abs(item[0] - batch_size))
            return closest_cost

        # Fallback 2: Estimate from forward costs C_T and C_D if available
        if self.forward_target and self.forward_draft:
            closest_b_target = min(self.forward_target.keys(), key=lambda b: abs(b - batch_size))
            c_t = self.forward_target[closest_b_target]
            c_d = self.forward_draft.get(closest_b_target, c_t * 0.25)
            if k == 0:
                return c_t
            c_v = c_t * (1.0 + 0.008 * (k - 1))
            return k * c_d + c_v

        # Fallback 3: Generic heuristic if profile is completely empty
        base_t = 20.0
        draft_t = 5.0
        return base_t if k == 0 else (base_t + k * draft_t)

    def get_forward_costs(self, batch_size: int = 1) -> Tuple[float, float]:
        """Returns (C_T, C_D) for the closest matching batch size."""
        if not self.forward_target or not self.forward_draft:
            c_vanilla = self.get_cost(k=0, batch_size=batch_size)
            return c_vanilla, c_vanilla * 0.25

        closest_b = min(self.forward_target.keys(), key=lambda b: abs(b - batch_size))
        return self.forward_target[closest_b], self.forward_draft[closest_b]

    def get_verification_cost(self, k: int, batch_size: int = 1) -> float:
        """Returns C_V(k, B) in milliseconds."""
        if (k, batch_size) in self.verification_costs:
            return self.verification_costs[(k, batch_size)]
        c_t, _ = self.get_forward_costs(batch_size=batch_size)
        return c_t * (1.0 + 0.008 * max(0, k - 1))

    @staticmethod
    def estimate_expected_tokens(k: int, alpha: float) -> float:
        """Calculates expected accepted tokens per step: E[tau] = 1 + sum_(i=1..k) (alpha^i)."""
        if k <= 0:
            return 1.0
        alpha_clamped = min(max(alpha, 0.0), 1.0)
        gain = 1.0
        term = 1.0
        for _ in range(k):
            term *= alpha_clamped
            gain += term
        return gain

    def calculate_speedup(self, k: int, alpha: float, batch_size: int = 1) -> float:
        """Calculates theoretical speedup relative to vanilla: Speedup = (E[tau] * C_vanilla) / C(k, B)."""
        if k == 0:
            return 1.0
        c_vanilla = self.get_cost(k=0, batch_size=batch_size)
        c_spec = self.get_cost(k=k, batch_size=batch_size)
        if c_spec <= 1e-6:
            return 1.0
        e_tau = self.estimate_expected_tokens(k=k, alpha=alpha)
        return (e_tau * c_vanilla) / c_spec

    def calculate_alpha_star(
        self,
        k: int,
        batch_size: int = 1,
        iterations: int = 60,
    ) -> Optional[float]:
        """Calculates the break-even acceptance rate alpha* in (0, 1) where Speedup == 1.0."""
        if k <= 0:
            return 0.0

        if (k, batch_size) in self.break_even_alpha:
            precomputed = self.break_even_alpha[(k, batch_size)]
            if precomputed is not None:
                return precomputed

        c_vanilla = self.get_cost(k=0, batch_size=batch_size)
        c_spec = self.get_cost(k=k, batch_size=batch_size)
        target_ratio = c_spec / max(c_vanilla, 1e-4)

        if target_ratio <= 1.0:
            return 0.0
        if 1.0 + k < target_ratio:
            return None

        # Bisection solver
        low, high = 0.0, 1.0
        for _ in range(iterations):
            mid = (low + high) / 2.0
            val = self.estimate_expected_tokens(k=k, alpha=mid)
            if val < target_ratio:
                low = mid
            else:
                high = mid
        return round((low + high) / 2.0, 4)

    def get_optimal_budget(
        self,
        alpha: float,
        batch_size: int = 1,
        candidate_budgets: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Finds k* in candidate_budgets maximizing speedup.
        
        Guarantees fallback to k=0 (Speedup = 1.0) if all candidate budgets produce
        negative speedup (< 1.0).
        """
        candidates = candidate_budgets or [0, 2, 3, 4]
        best_k = 0
        best_speedup = 1.0

        for cand_k in candidates:
            if cand_k == 0:
                continue
            speedup = self.calculate_speedup(k=cand_k, alpha=alpha, batch_size=batch_size)
            if speedup > best_speedup:
                best_speedup = speedup
                best_k = cand_k

        return best_k, best_speedup

    def to_cost_profile(self) -> CostProfile:
        """Converts to a CostProfile dataclass for drop-in use in BudgetController."""
        return CostProfile(matrix=dict(self.matrix))
