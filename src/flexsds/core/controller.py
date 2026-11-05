"""TreeAdaptive controller module with hardware-aware dynamic thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

try:
    from flexsds.core.cost_model import FlexCostModel
except ModuleNotFoundError:
    from src.flexsds.core.cost_model import FlexCostModel


@dataclass
class TreeAdaptiveDecision:
    """Represents an adaptive budget decision made by TreeAdaptiveController."""
    k: int
    alpha_effective: float
    alpha_star: float
    expected_tokens: float
    reason: str


class TreeAdaptiveController:
    """Hardware-aware TreeAdaptive Speculative Controller v0.2.

    Dynamically solves break-even acceptance rate alpha*(k, B) against the GPU cost model
    and adapts tree depth/budget k in candidate_budgets to prevent negative speedup under high load.
    """

    def __init__(
        self,
        cost_model: Optional[FlexCostModel] = None,
        window_size: int = 10,
        ema_beta: float = 0.8,
        initial_alpha: float = 0.8,
        candidate_budgets: Optional[List[int]] = None,
        min_free_vram_mb: float = 1500.0,
        probe_interval: int = 20,
        probe_k: int = 2,
        kv_cache_high_pct: float = 90.0,
        kv_cache_recover_pct: float = 80.0,
    ):
        self.cost_model = cost_model or FlexCostModel()
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.alpha_ema = initial_alpha
        self.initial_alpha = initial_alpha
        self.candidate_budgets = candidate_budgets or [0, 2, 3, 4]
        self.min_free_vram_mb = min_free_vram_mb
        self.probe_interval = probe_interval
        self.probe_k = probe_k
        self.kv_cache_high_pct = kv_cache_high_pct
        self.kv_cache_recover_pct = kv_cache_recover_pct

        self.step_count = 0
        self.consecutive_fallback_steps = 0
        self._memory_pressure_active = False
        self._window: deque[float] = deque(maxlen=window_size)
        self._history: List[dict] = []

    def get_effective_alpha(self) -> float:
        """Returns unbiased alpha taking into account warm-up correction."""
        if self.step_count == 0:
            return self.initial_alpha

        beta_power = self.ema_beta ** self.step_count
        denominator = 1.0 - beta_power
        if denominator <= 1e-6:
            return self.alpha_ema

        unbiased = (self.alpha_ema - self.initial_alpha * beta_power) / denominator
        return min(max(unbiased, 0.0), 1.0)

    def update(self, num_accepted: int, num_drafted: int) -> float:
        """Updates rolling alpha telemetry from the latest step verification."""
        self.step_count += 1
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
        kv_cache_usage_pct: float = 0.0,
    ) -> TreeAdaptiveDecision:
        """Selects optimal budget k* using dynamic thresholds alpha*(k, B)."""
        effective_alpha = self.get_effective_alpha()

        # 1. Memory Pressure Rule (OOM protection)
        if free_vram_mb < self.min_free_vram_mb or kv_cache_usage_pct > self.kv_cache_high_pct:
            self._memory_pressure_active = True
            decision = TreeAdaptiveDecision(
                k=0,
                alpha_effective=effective_alpha,
                alpha_star=0.0,
                expected_tokens=1.0,
                reason=f"Memory pressure (free VRAM: {free_vram_mb:.0f}MB, KV: {kv_cache_usage_pct:.1f}%)",
            )
            self._history.append({"k": 0, "alpha": effective_alpha, "reason": decision.reason})
            return decision

        if self._memory_pressure_active:
            if kv_cache_usage_pct < self.kv_cache_recover_pct:
                self._memory_pressure_active = False
            else:
                decision = TreeAdaptiveDecision(
                    k=0,
                    alpha_effective=effective_alpha,
                    alpha_star=0.0,
                    expected_tokens=1.0,
                    reason=f"Memory cooldown (KV: {kv_cache_usage_pct:.1f}% >= {self.kv_cache_recover_pct}%)",
                )
                self._history.append({"k": 0, "alpha": effective_alpha, "reason": decision.reason})
                return decision

        # 2. Dynamic Threshold Solution: Solve break-even alpha*(k_min, B)
        # Find minimum non-zero candidate budget (e.g. k=2)
        nonzero_cands = [k for k in self.candidate_budgets if k > 0]
        min_k = min(nonzero_cands) if nonzero_cands else 2
        alpha_star = self.cost_model.calculate_alpha_star(k=min_k, batch_size=batch_size)

        # Fallback Check against Dynamic Break-Even
        if effective_alpha < alpha_star:
            self.consecutive_fallback_steps += 1
            if self.probe_interval > 0 and self.consecutive_fallback_steps >= self.probe_interval:
                self.consecutive_fallback_steps = 0
                decision = TreeAdaptiveDecision(
                    k=self.probe_k,
                    alpha_effective=effective_alpha,
                    alpha_star=alpha_star,
                    expected_tokens=self.cost_model.estimate_expected_tokens(self.probe_k, effective_alpha),
                    reason=f"Dynamic probe k={self.probe_k} (alpha {effective_alpha:.3f} < alpha* {alpha_star:.3f})",
                )
                self._history.append({"k": self.probe_k, "alpha": effective_alpha, "reason": decision.reason})
                return decision

            decision = TreeAdaptiveDecision(
                k=0,
                alpha_effective=effective_alpha,
                alpha_star=alpha_star,
                expected_tokens=1.0,
                reason=f"Dynamic fallback: alpha {effective_alpha:.3f} < alpha* {alpha_star:.3f} at B={batch_size}",
            )
            self._history.append({"k": 0, "alpha": effective_alpha, "reason": decision.reason})
            return decision

        self.consecutive_fallback_steps = 0

        # 3. Optimal Budget Argmax over Hardware Cost Model
        best_k, best_speedup = self.cost_model.select_optimal_budget(
            alpha=effective_alpha,
            batch_size=batch_size,
            candidate_budgets=self.candidate_budgets,
        )

        e_tau = self.cost_model.estimate_expected_tokens(k=best_k, alpha=effective_alpha)
        decision = TreeAdaptiveDecision(
            k=best_k,
            alpha_effective=effective_alpha,
            alpha_star=alpha_star,
            expected_tokens=e_tau,
            reason=f"Optimal budget k={best_k} with theoretical speedup {best_speedup:.2f}x at B={batch_size}",
        )
        self._history.append({"k": best_k, "alpha": effective_alpha, "reason": decision.reason})
        return decision
