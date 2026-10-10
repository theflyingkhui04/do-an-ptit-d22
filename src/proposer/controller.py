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


class BudgetController:
    """Adaptive Speculative Controller v0 based on rolling acceptance rate alpha."""

    def __init__(
        self,
        window_size: int = 10,
        ema_beta: float = 0.8,
        initial_alpha: float = 0.8,
        fallback_threshold: float = 0.65,
        high_threshold: float = 0.80,
        candidate_budgets: Optional[List[int]] = None,
        cost_profile: Optional[CostProfile] = None,
        min_free_vram_mb: float = 1500.0,
    ):
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.alpha_ema = initial_alpha
        self.fallback_threshold = fallback_threshold
        self.high_threshold = high_threshold
        self.candidate_budgets = candidate_budgets or [0, 2, 3, 4]
        self.cost_profile = cost_profile or CostProfile()
        self.min_free_vram_mb = min_free_vram_mb

        self._window: deque[float] = deque(maxlen=window_size)
        self._history: List[dict] = []

    def get_rolling_alpha(self) -> float:
        """Returns the current Exponential Moving Average of alpha."""
        return self.alpha_ema

    def estimate_expected_tokens(self, k: int, alpha: float) -> float:
        """Calculates expected token generation per verification step: E[tau] = 1 + sum_(i=1..k) (alpha^i)."""
        if k <= 0:
            return 1.0
        gain = 1.0
        curr_power = 1.0
        for _ in range(k):
            curr_power *= alpha
            gain += curr_power
        return gain

    def update(self, num_accepted: int, num_drafted: int) -> float:
        """Updates rolling alpha telemetry from the latest step verification."""
        if num_drafted <= 0:
            step_alpha = self.alpha_ema
        else:
            step_alpha = min(max(num_accepted / num_drafted, 0.0), 1.0)

        self._window.append(step_alpha)
        self.alpha_ema = self.ema_beta * self.alpha_ema + (1.0 - self.ema_beta) * step_alpha
        return self.alpha_ema

    def select_budget(
        self,
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
    ) -> BudgetDecision:
        """Selects optimal budget k* in candidate_budgets based on VRAM, alpha, and cost model."""
        alpha = self.alpha_ema

        # 1. Memory Pressure Rule (Chống OOM)
        if free_vram_mb < self.min_free_vram_mb:
            decision = BudgetDecision(
                k=0,
                alpha_ema=alpha,
                expected_tokens=1.0,
                reason=f"VRAM pressure: free {free_vram_mb:.1f}MB < min {self.min_free_vram_mb:.1f}MB",
            )
            self._history.append({"k": 0, "alpha": alpha, "reason": decision.reason})
            return decision

        # 2. Safety Fallback Gate (Triệt tiêu tăng tốc âm khi alpha thấp)
        if alpha < self.fallback_threshold:
            decision = BudgetDecision(
                k=0,
                alpha_ema=alpha,
                expected_tokens=1.0,
                reason=f"Low acceptance fallback: alpha {alpha:.3f} < threshold {self.fallback_threshold:.3f}",
            )
            self._history.append({"k": 0, "alpha": alpha, "reason": decision.reason})
            return decision

        # 3. Cost-Model Argmax kết hợp Heuristic Guards
        c_vanilla = self.cost_profile.get_cost(k=0, batch_size=batch_size)
        vanilla_efficiency = 1.0 / max(c_vanilla, 1e-4)

        best_k = 0
        best_efficiency = vanilla_efficiency
        best_expected_tokens = 1.0

        for cand_k in self.candidate_budgets:
            if cand_k == 0:
                continue

            # Heuristic guard: Không mở rộng k=4 trừ khi alpha >= high_threshold (0.80)
            if cand_k == 4 and alpha < self.high_threshold:
                continue

            e_tau = self.estimate_expected_tokens(cand_k, alpha)
            c_cand = self.cost_profile.get_cost(k=cand_k, batch_size=batch_size)
            efficiency = e_tau / max(c_cand, 1e-4)

            # Heuristic mid-alpha: nếu alpha < high_threshold (0.80) và cand_k == 2,
            # ưu tiên k=2 hơn k=3 nếu hiệu năng ngang ngửa để giảm tải tính toán
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_k = cand_k
                best_expected_tokens = e_tau

        reason = (
            "Vanilla fallback (efficiency bound)"
            if best_k == 0
            else f"Optimal budget k={best_k} with eff={best_efficiency:.4f}"
        )

        decision = BudgetDecision(
            k=best_k,
            alpha_ema=alpha,
            expected_tokens=best_expected_tokens,
            reason=reason,
        )
        self._history.append({"k": best_k, "alpha": alpha, "reason": decision.reason})
        return decision

    def reset(self, initial_alpha: Optional[float] = None) -> None:
        """Resets controller state when beginning a new sequence/prompt."""
        if initial_alpha is not None:
            self.alpha_ema = initial_alpha
        self._window.clear()
        self._history.clear()

    def get_stats(self) -> dict:
        """Exports runtime telemetry matching Interface A contract."""
        tree_budgets = [h["k"] for h in self._history]
        return {
            "alpha_ema": self.alpha_ema,
            "tree_budgets": tree_budgets,
            "step_count": len(self._window),
            "mean_k": (sum(tree_budgets) / len(tree_budgets)) if tree_budgets else 0.0,
            "history": list(self._history),
        }



