"""Hardware cost model loader and dynamic break-even solver for FlexSDS."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from flexsds.core.hardware import HardwareInfo
except ModuleNotFoundError:
    from src.flexsds.core.hardware import HardwareInfo

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class FlexCostModel:
    """Hardware cost model evaluating forward latencies and dynamic break-even alpha*(k, B)."""

    def __init__(
        self,
        cost_matrix: Optional[Dict[Tuple[int, int], float]] = None,
        forward_target: Optional[Dict[int, float]] = None,
        forward_draft: Optional[Dict[int, float]] = None,
        metadata: Optional[dict] = None,
    ):
        self.matrix = cost_matrix or {}
        self.forward_target = forward_target or {}
        self.forward_draft = forward_draft or {}
        self.metadata = metadata or {}

    @classmethod
    def from_json(cls, path_or_dict: Union[str, Path, dict]) -> FlexCostModel:
        """Loads cost model from a profile JSON file."""
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        matrix: Dict[Tuple[int, int], float] = {}
        for key_str, val in data.get("cost_matrix_ms", {}).items():
            parts = key_str.split("_")
            if len(parts) == 2 and parts[0].startswith("k") and parts[1].startswith("b"):
                k_val = int(parts[0][1:])
                b_val = int(parts[1][1:])
                matrix[(k_val, b_val)] = float(val)

        c_target: Dict[int, float] = {}
        f_costs = data.get("forward_costs_median_ms", {})
        for b_str, val in f_costs.get("target_c_t", {}).items():
            c_target[int(b_str)] = float(val)

        c_draft: Dict[int, float] = {}
        for b_str, val in f_costs.get("draft_c_d", {}).items():
            c_draft[int(b_str)] = float(val)

        return cls(
            cost_matrix=matrix,
            forward_target=c_target,
            forward_draft=c_draft,
            metadata=data.get("metadata", {}),
        )

    def get_forward_costs(self, batch_size: int = 1) -> Tuple[float, float]:
        """Returns (C_T, C_D) in ms for the closest matching batch size."""
        if not self.forward_target:
            return 20.0, 5.0
        closest_b = min(self.forward_target.keys(), key=lambda b: abs(b - batch_size))
        c_t = self.forward_target[closest_b]
        c_d = self.forward_draft.get(closest_b, c_t * 0.25)
        return c_t, c_d

    def get_step_cost(self, k: int, batch_size: int = 1) -> float:
        """Gets step execution latency C(k, B) in milliseconds."""
        if (k, batch_size) in self.matrix:
            return self.matrix[(k, batch_size)]
        c_t, c_d = self.get_forward_costs(batch_size=batch_size)
        if k == 0:
            return c_t
        attn_verify_factor = float(self.metadata.get("attn_verify_factor", 0.008))
        c_v = c_t * (1.0 + attn_verify_factor * max(0, k - 1))
        return k * c_d + c_v

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

    def calculate_alpha_star(
        self,
        k: int,
        batch_size: int = 1,
        iterations: int = 50,
    ) -> float:
        """Solves break-even acceptance rate alpha*(k, B) where Speedup == 1.0."""
        if k <= 0:
            return 0.0
        c_vanilla = self.get_step_cost(k=0, batch_size=batch_size)
        c_spec = self.get_step_cost(k=k, batch_size=batch_size)
        target_ratio = c_spec / max(c_vanilla, 1e-4)

        if target_ratio <= 1.0:
            return 0.0
        if 1.0 + k < target_ratio:
            return 1.0

        low, high = 0.0, 1.0
        for _ in range(iterations):
            mid = (low + high) / 2.0
            val = self.estimate_expected_tokens(k=k, alpha=mid)
            if val < target_ratio:
                low = mid
            else:
                high = mid
        return round((low + high) / 2.0, 4)

    def select_optimal_budget(
        self,
        alpha: float,
        batch_size: int = 1,
        candidate_budgets: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Finds k* maximizing speedup; falls back to 0 if all k yield speedup < 1.0."""
        candidates = candidate_budgets or [0, 2, 3, 4]
        c_vanilla = self.get_step_cost(k=0, batch_size=batch_size)
        best_k = 0
        best_speedup = 1.0

        for cand_k in candidates:
            if cand_k == 0:
                continue
            c_cand = self.get_step_cost(k=cand_k, batch_size=batch_size)
            e_tau = self.estimate_expected_tokens(k=cand_k, alpha=alpha)
            speedup = (e_tau * c_vanilla) / max(c_cand, 1e-4)
            if speedup > best_speedup:
                best_speedup = speedup
                best_k = cand_k

        return best_k, round(best_speedup, 3)


def load_profile_for_hardware(hw_info: Optional[HardwareInfo] = None) -> Tuple[FlexCostModel, str]:
    """Auto-detects GPU architecture and loads corresponding pre-computed profile."""
    sm_arch = hw_info.sm_architecture if hw_info else "sm_89"
    
    # Try exact match by sm_architecture
    arch_map = {
        "sm_89": "profile_sm89_l4.json",
        "sm_75": "profile_sm75_t4.json",
    }

    profile_filename = arch_map.get(sm_arch, "profile_sm89_l4.json")
    profile_path = PROFILES_DIR / profile_filename

    if profile_path.exists():
        return FlexCostModel.from_json(profile_path), profile_filename

    # Fallback to analytical default
    default_target = {1: 20.0, 2: 20.5, 4: 21.0, 8: 22.0, 16: 25.0, 32: 32.0}
    default_draft = {1: 5.0, 2: 5.2, 4: 5.4, 8: 5.8, 16: 6.5, 32: 7.2}
    return FlexCostModel(forward_target=default_target, forward_draft=default_draft), "analytical_fallback"
