"""BudgetController module for Adaptive Speculative Decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from collections import deque


@dataclass
class BudgetDecision:
    """Represents a budget selection decision made by BudgetController."""
    k: int
    alpha_ema: float
    expected_tokens: float
    reason: str


@dataclass
class CostProfile:
    """Hardware cost profile lookup table C(k, B) in milliseconds."""
    matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def get_cost(self, k: int, batch_size: int = 1) -> float:
        """Get step latency in ms for draft depth k and batch size.
        
        Falls back to matching k if exact batch is missing, or a linear heuristic.
        """
        if (k, batch_size) in self.matrix:
            return self.matrix[(k, batch_size)]

        # Fallback 1: Tìm key có cùng k
        matching_k = [cost for (cand_k, _), cost in self.matrix.items() if cand_k == k]
        if matching_k:
            return matching_k[0]

        # Fallback 2: Ước lượng tuyến tính mặc định nếu chưa có profile
        base_t = 20.0  # ms for vanilla step
        draft_step_t = 3.0  # ms for draft step
        return base_t if k == 0 else (base_t + k * draft_step_t)
