"""BudgetController module for Adaptive Speculative Decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Tuple, List, Optional
from collections import deque


@dataclass
class BudgetDecision:
    """Represents a budget selection decision made by BudgetController."""
    k: int
    alpha_ema: float
    expected_tokens: float
    reason: str
    speedup: Optional[float] = None
    alpha_star: Optional[float] = None
    status: str = "ok"
    decision_policy: str = "threshold"
    cost_k_ms: Optional[float] = None
    cost_baseline_ms: Optional[float] = None
    cost_ref_ms: Optional[float] = None
    non_monotonic_cost: bool = False


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


class EmpiricalCostProfile(CostProfile):
    """Use the closest measured batch for an observed K; never invent costs."""
    def get_cost(self, k: int, batch_size: int = 1) -> float:
        batches = [b for cand_k, b in self.matrix if cand_k == k]
        if not batches:
            raise ValueError(f'No measured cost for K={k}')
        closest = min(batches, key=lambda b: (abs(b-batch_size), b))
        return self.matrix[(k, closest)]


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
        history_limit: Optional[int] = None,
    ):
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.alpha_ema = initial_alpha
        self.fallback_threshold = fallback_threshold
        self.high_threshold = high_threshold
        self.candidate_budgets = candidate_budgets or [0, 2, 3, 4]
        self.cost_profile = cost_profile or CostProfile()
        self.min_free_vram_mb = min_free_vram_mb

        self.initial_alpha = initial_alpha
        self._window: deque[float] = deque(maxlen=window_size)
        if history_limit is not None and (type(history_limit) is not int or history_limit < 1):
            raise ValueError("history_limit must be a positive integer")
        self._history = [] if history_limit is None else deque(maxlen=history_limit)

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
        kv_cache_usage_pct: float = 0.0,
    ) -> BudgetDecision:
        """Selects optimal budget k* in candidate_budgets based on VRAM, alpha, and cost model.

        kv_cache_usage_pct is accepted for interface parity with AdaptiveDraftController
        and is not used by the v0 base controller.
        """
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
            self.initial_alpha = initial_alpha
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


class AdaptiveDraftController(BudgetController):
    """Runtime Adaptive Draft Controller for vLLM speculative decoding.

    Extends BudgetController with:
    - 10-step rolling window tracking raw acceptance samples
    - Unbiased EMA warm-up correction for fast reaction to low acceptance rate
    - Modular budget rules (co giãn k in {0, 2, 3, 4})
    - Memory pressure rule (KV-cache usage + VRAM guard) with cooldown and hysteresis
    - Self-diagnostic window statistics (mean, variance, min, max, is_warmed_up)
    """

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
        probe_interval: int = 20,
        probe_k: int = 2,
        kv_cache_high_pct: float = 90.0,
        kv_cache_recover_pct: float = 80.0,
        memory_cooldown_steps: int = 3,
        enable_vram_guard: bool = True,
        min_speedup_ratio: float = 1.05,
        history_limit: Optional[int] = None,
    ):
        super().__init__(
            window_size=window_size,
            ema_beta=ema_beta,
            initial_alpha=initial_alpha,
            fallback_threshold=fallback_threshold,
            high_threshold=high_threshold,
            candidate_budgets=candidate_budgets,
            cost_profile=cost_profile,
            min_free_vram_mb=min_free_vram_mb,
            history_limit=history_limit,
        )
        if kv_cache_recover_pct > kv_cache_high_pct:
            raise ValueError(
                f"kv_cache_recover_pct ({kv_cache_recover_pct}) must be <= "
                f"kv_cache_high_pct ({kv_cache_high_pct})"
            )
        if memory_cooldown_steps < 1:
            raise ValueError("memory_cooldown_steps must be >= 1")
        if not (isinstance(min_speedup_ratio, (int, float)) and math.isfinite(min_speedup_ratio) and min_speedup_ratio >= 1.0):
            raise ValueError("min_speedup_ratio must be finite and >= 1.0")

        self.step_count = 0
        self.probe_interval = probe_interval
        self.probe_k = probe_k
        self.consecutive_fallback_steps = 0
        self.kv_cache_high_pct = kv_cache_high_pct
        self.kv_cache_recover_pct = kv_cache_recover_pct
        self.memory_cooldown_steps = memory_cooldown_steps
        self.enable_vram_guard = enable_vram_guard
        self.min_speedup_ratio = float(min_speedup_ratio)

        # Hysteresis state: steps spent at k=0 since the last memory trigger
        self._memory_pressure_active = False
        self._memory_pressure_steps = 0

    def update(self, num_accepted: int, num_drafted: int) -> float:
        """Updates rolling window and EMA telemetry from the latest step verification."""
        self.step_count += 1
        return super().update(num_accepted=num_accepted, num_drafted=num_drafted)

    def get_window_samples(self) -> List[float]:
        """Returns the list of raw alpha samples in the current sliding window."""
        return list(self._window)

    def get_effective_alpha(self) -> float:
        """Returns unbiased alpha taking into account warm-up correction.

        If step_count == 0, returns initial_alpha.
        When step_count > 0:
        unbiased_alpha = (alpha_ema - initial_alpha * (ema_beta ** step_count)) / (1.0 - (ema_beta ** step_count))
        Clamped to [0.0, 1.0].
        """
        if self.step_count == 0:
            return self.initial_alpha

        beta_power = self.ema_beta ** self.step_count
        denominator = 1.0 - beta_power
        if denominator <= 1e-6:
            return self.alpha_ema

        unbiased = (self.alpha_ema - self.initial_alpha * beta_power) / denominator
        return min(max(unbiased, 0.0), 1.0)

    def get_window_stats(self) -> dict:
        """Returns self-diagnostic statistics of the rolling window."""
        samples = list(self._window)
        is_warmed_up = self.step_count >= self.window_size

        if not samples:
            return {
                "mean": self.initial_alpha,
                "min": self.initial_alpha,
                "max": self.initial_alpha,
                "variance": 0.0,
                "is_warmed_up": False,
                "samples": [],
                "step_count": self.step_count,
            }

        mean_val = sum(samples) / len(samples)
        var_val = sum((x - mean_val) ** 2 for x in samples) / len(samples)

        return {
            "mean": mean_val,
            "min": min(samples),
            "max": max(samples),
            "variance": var_val,
            "is_warmed_up": is_warmed_up,
            "samples": samples,
            "step_count": self.step_count,
        }

    def apply_memory_pressure_rule(
        self,
        free_vram_mb: float,
        kv_cache_usage_pct: float = 0.0,
    ) -> Optional[Tuple[int, str]]:
        """Memory pressure rule with hysteresis, evaluated once per decode step.

        - Trigger: KV-cache usage > kv_cache_high_pct, or (VRAM guard enabled)
          free VRAM < min_free_vram_mb -> force k=0 and (re)start cooldown.
        - Hold: k=0 for at least memory_cooldown_steps steps (trigger step included).
        - Release: after cooldown, only when KV-cache usage < kv_cache_recover_pct
          and free VRAM is back above the minimum; Budget Rule then takes over.

        kv_cache_usage_pct is in percent (0-100), e.g. 100 * scheduler.get_kv_cache_usage().
        Returns (k, reason) while memory pressure holds k=0, else None.
        """
        vram_low = self.enable_vram_guard and free_vram_mb < self.min_free_vram_mb
        kv_high = kv_cache_usage_pct > self.kv_cache_high_pct

        if kv_high or vram_low:
            self._memory_pressure_active = True
            self._memory_pressure_steps = 1
            if kv_high:
                reason = (
                    f"KV-cache pressure: usage {kv_cache_usage_pct:.1f}% > "
                    f"high {self.kv_cache_high_pct:.1f}%"
                )
            else:
                reason = (
                    f"VRAM pressure: free {free_vram_mb:.1f}MB < "
                    f"min {self.min_free_vram_mb:.1f}MB"
                )
            return (0, reason)

        if not self._memory_pressure_active:
            return None

        cooldown_done = self._memory_pressure_steps >= self.memory_cooldown_steps
        if cooldown_done and kv_cache_usage_pct < self.kv_cache_recover_pct:
            self._memory_pressure_active = False
            self._memory_pressure_steps = 0
            return None

        self._memory_pressure_steps += 1
        if not cooldown_done:
            reason = (
                f"Memory pressure cooldown: step {self._memory_pressure_steps}/"
                f"{self.memory_cooldown_steps}"
            )
        else:
            reason = (
                f"Memory pressure hold: KV-cache {kv_cache_usage_pct:.1f}% >= "
                f"recover {self.kv_cache_recover_pct:.1f}%"
            )
        return (0, reason)

    def apply_budget_rule(self, alpha: float) -> Tuple[int, str]:
        """Core budget rule adapting k in {0, 2, 3, 4} based on empirical alpha thresholds.

        - alpha >= high_threshold (0.80): maintain k=3 (or expand to k=4 if alpha >= 0.90)
        - fallback_threshold <= alpha < high_threshold: throttle down to k=2
        - alpha < fallback_threshold (0.65): fallback to k=0 with periodic probing (k=probe_k every probe_interval steps)
        """
        if alpha < self.fallback_threshold:
            self.consecutive_fallback_steps += 1
            if self.probe_interval > 0 and self.consecutive_fallback_steps >= self.probe_interval:
                # Kích hoạt probing để thoát bẫy kẹt Vanilla vĩnh viễn
                self.consecutive_fallback_steps = 0
                return (
                    self.probe_k,
                    f"Periodic probe: reached {self.probe_interval} fallback steps (alpha={alpha:.3f}), probing with k={self.probe_k}",
                )
            hold_str = f" (hold {self.consecutive_fallback_steps}/{self.probe_interval})" if self.probe_interval > 0 else ""
            return (
                0,
                f"Low acceptance fallback: alpha {alpha:.3f} < threshold {self.fallback_threshold:.3f}{hold_str}",
            )

        # alpha >= self.fallback_threshold -> reset bộ đếm fallback
        self.consecutive_fallback_steps = 0

        if alpha >= self.high_threshold:
            if alpha >= 0.90 and 4 in self.candidate_budgets:
                return (4, f"High acceptance (aggressive): alpha {alpha:.3f} >= 0.90, selected k=4")
            return (3, f"High acceptance: alpha {alpha:.3f} >= threshold {self.high_threshold:.3f}, maintain k=3")

        # fallback_threshold <= alpha < high_threshold
        return (
            2,
            f"Mid acceptance: alpha {alpha:.3f} in [{self.fallback_threshold:.2f}, {self.high_threshold:.2f}), throttled to k=2",
        )

    def solve_break_even_alpha(
        self,
        k: int,
        cost_ratio: float,
        min_speedup_ratio: Optional[float] = None,
    ) -> Optional[float]:
        """Solves E[tau(k, alpha)] = min_speedup_ratio * cost_ratio on [0, 1]."""
        if k <= 0:
            return None
        threshold = self.min_speedup_ratio if min_speedup_ratio is None else min_speedup_ratio
        target = threshold * cost_ratio
        if not math.isfinite(target):
            return None
        if target <= 1.0:
            return 0.0
        if self.estimate_expected_tokens(k, 1.0) < target:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.estimate_expected_tokens(k, mid) < target:
                lo = mid
            else:
                hi = mid
        return hi

    def _select_probe_k(self, positive_candidates: List[int]) -> int:
        """Selects a valid positive probe budget contained in candidate_budgets."""
        if not positive_candidates:
            return 0
        if self.probe_k in positive_candidates:
            return self.probe_k
        le_candidates = [k for k in positive_candidates if k <= self.probe_k]
        if le_candidates:
            return max(le_candidates)
        return min(positive_candidates)

    def apply_cost_driven_budget_rule(
        self,
        alpha: float,
        batch_size: int = 1,
    ) -> BudgetDecision:
        """Selects k* using measured C(k, B) with lower-bound C_ref(B) normalization."""
        positive_candidates = sorted(k for k in set(self.candidate_budgets) if k > 0)
        all_candidates = [0] + positive_candidates

        costs: Dict[int, float] = {}
        for cand_k in all_candidates:
            cost_val = self.cost_profile.get_cost(k=cand_k, batch_size=batch_size)
            if not (isinstance(cost_val, (int, float)) and math.isfinite(cost_val) and cost_val > 0.0):
                raise ValueError(
                    f"Cost profile entry for k={cand_k}, batch_size={batch_size} must be finite and positive; got {cost_val!r}"
                )
            costs[cand_k] = float(cost_val)

        c_baseline = costs[0]
        c_ref = min(costs.values())
        non_monotonic = c_ref < c_baseline

        best_k = 0
        best_speedup = 1.0
        best_expected_tokens = 1.0

        for cand_k in positive_candidates:
            e_tau = self.estimate_expected_tokens(cand_k, alpha)
            speedup = e_tau * (c_ref / costs[cand_k])
            if speedup >= self.min_speedup_ratio and speedup > best_speedup + 1e-6:
                best_k = cand_k
                best_speedup = speedup
                best_expected_tokens = e_tau

        if best_k > 0:
            self.consecutive_fallback_steps = 0
            alpha_star = self.solve_break_even_alpha(best_k, costs[best_k] / c_ref)
            reason = (
                f"Cost-driven budget k={best_k} (B={batch_size}, alpha={alpha:.3f}, "
                f"C_k={costs[best_k]:.2f}ms, C_ref={c_ref:.2f}ms, speedup={best_speedup:.3f}x)"
            )
            return BudgetDecision(
                k=best_k,
                alpha_ema=alpha,
                expected_tokens=best_expected_tokens,
                reason=reason,
                speedup=best_speedup,
                alpha_star=alpha_star,
                status="ok",
                decision_policy="cost_model",
                cost_k_ms=costs[best_k],
                cost_baseline_ms=c_baseline,
                cost_ref_ms=c_ref,
                non_monotonic_cost=non_monotonic,
            )

        feasible_stars = [
            star
            for cand_k in positive_candidates
            if (star := self.solve_break_even_alpha(cand_k, costs[cand_k] / c_ref)) is not None
        ]
        fallback_alpha_star = min(feasible_stars) if feasible_stars else None

        self.consecutive_fallback_steps += 1
        if self.probe_interval > 0 and self.consecutive_fallback_steps >= self.probe_interval:
            self.consecutive_fallback_steps = 0
            probe_k = self._select_probe_k(positive_candidates)
            if probe_k > 0:
                probe_e_tau = self.estimate_expected_tokens(probe_k, alpha)
                probe_speedup = probe_e_tau * (c_ref / costs[probe_k])
                reason = (
                    f"Periodic probe (cost-driven): reached {self.probe_interval} fallback steps "
                    f"(B={batch_size}, alpha={alpha:.3f}), probing with k={probe_k}"
                )
                return BudgetDecision(
                    k=probe_k,
                    alpha_ema=alpha,
                    expected_tokens=probe_e_tau,
                    reason=reason,
                    speedup=probe_speedup,
                    alpha_star=self.solve_break_even_alpha(probe_k, costs[probe_k] / c_ref),
                    status="probing",
                    decision_policy="cost_model",
                    cost_k_ms=costs[probe_k],
                    cost_baseline_ms=c_baseline,
                    cost_ref_ms=c_ref,
                    non_monotonic_cost=non_monotonic,
                )

        hold_str = (
            f" (hold {self.consecutive_fallback_steps}/{self.probe_interval})"
            if self.probe_interval > 0
            else ""
        )
        reason = (
            f"Cost-model fallback: no k>0 achieves speedup >= {self.min_speedup_ratio:.2f}x "
            f"(B={batch_size}, alpha={alpha:.3f}, C_0={c_baseline:.2f}ms, C_ref={c_ref:.2f}ms){hold_str}"
        )
        return BudgetDecision(
            k=0,
            alpha_ema=alpha,
            expected_tokens=1.0,
            reason=reason,
            speedup=1.0,
            alpha_star=fallback_alpha_star,
            status="fallback_gate",
            decision_policy="cost_model",
            cost_k_ms=c_baseline,
            cost_baseline_ms=c_baseline,
            cost_ref_ms=c_ref,
            non_monotonic_cost=non_monotonic,
        )

    def select_budget(
        self,
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
        kv_cache_usage_pct: float = 0.0,
    ) -> BudgetDecision:
        """Selects optimal budget k* using unbiased alpha, memory guard, and hardware cost profile."""
        effective_alpha = self.get_effective_alpha()

        # 1. Memory Pressure Rule (Chống OOM — ưu tiên cao nhất)
        mem_decision = self.apply_memory_pressure_rule(
            free_vram_mb=free_vram_mb,
            kv_cache_usage_pct=kv_cache_usage_pct,
        )
        if mem_decision is not None:
            k, reason = mem_decision
            decision = BudgetDecision(
                k=k,
                alpha_ema=effective_alpha,
                expected_tokens=1.0,
                reason=reason,
                speedup=1.0,
                status="memory_guard",
                decision_policy="memory_guard",
            )
            self._history.append(
                {
                    "k": k,
                    "alpha": effective_alpha,
                    "reason": reason,
                    "decision_policy": decision.decision_policy,
                    "speedup": decision.speedup,
                    "cost_k_ms": decision.cost_k_ms,
                }
            )
            return decision

        # 2. Hardware Cost-Driven Rule (khi có ma trận chi phí thực nghiệm)
        if bool(getattr(self.cost_profile, "matrix", None)):
            decision = self.apply_cost_driven_budget_rule(
                alpha=effective_alpha,
                batch_size=batch_size,
            )
            self._history.append(
                {
                    "k": decision.k,
                    "alpha": effective_alpha,
                    "reason": decision.reason,
                    "decision_policy": decision.decision_policy,
                    "speedup": decision.speedup,
                    "cost_k_ms": decision.cost_k_ms,
                }
            )
            return decision

        # 3. Legacy Threshold Rule (chỉ khi không có ma trận chi phí trong cost_profile)
        k, reason = self.apply_budget_rule(alpha=effective_alpha)
        e_tau = self.estimate_expected_tokens(k, effective_alpha)
        status = "probing" if "periodic probe" in reason.lower() else ("fallback_gate" if k == 0 else "ok")

        decision = BudgetDecision(
            k=k,
            alpha_ema=effective_alpha,
            expected_tokens=e_tau,
            reason=reason,
            status=status,
            decision_policy="threshold",
        )
        self._history.append(
            {
                "k": k,
                "alpha": effective_alpha,
                "reason": reason,
                "decision_policy": decision.decision_policy,
                "speedup": decision.speedup,
                "cost_k_ms": decision.cost_k_ms,
            }
        )
        return decision

    def reset(self, initial_alpha: Optional[float] = None) -> None:
        """Resets controller state when beginning a new sequence/prompt."""
        super().reset(initial_alpha=initial_alpha)
        if initial_alpha is not None:
            self.initial_alpha = initial_alpha
        self.step_count = 0
        self.consecutive_fallback_steps = 0
        self._memory_pressure_active = False
        self._memory_pressure_steps = 0

    def get_stats(self) -> dict:
        """Exports runtime telemetry matching Interface A contract plus adaptive diagnostics."""
        base_stats = super().get_stats()
        base_stats["effective_alpha"] = self.get_effective_alpha()
        base_stats["is_warmed_up"] = self.step_count >= self.window_size
        base_stats["step_count"] = self.step_count
        base_stats["consecutive_fallback_steps"] = self.consecutive_fallback_steps
        base_stats["memory_pressure_active"] = self._memory_pressure_active
        return base_stats




