"""TreeAdaptiveController v0.2 module with 2D tree topologies and dual-tier safety guards."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

try:
    from flexsds.core.cost_model import FlexCostModel, SolverResult
    from flexsds.core.tree_topology import get_tree_topology, TreeTopology
except ModuleNotFoundError:
    from src.flexsds.core.cost_model import FlexCostModel, SolverResult
    from src.flexsds.core.tree_topology import get_tree_topology, TreeTopology

from src.proposer.controller import BudgetController, BudgetDecision

logger = logging.getLogger(__name__)

# Value threshold representing missing/sentinel VRAM readings (e.g. 1e9 from server planners)
SENTINEL_VRAM_THRESHOLD_MB = 1.0e8



@dataclass
class TreeAdaptiveDecision(BudgetDecision):
    """Represents an adaptive tree budget decision made by TreeAdaptiveController v0.2."""
    topology_name: str = "T0"
    num_nodes: int = 0
    alpha_effective: float = 0.8
    alpha_star: Optional[float] = None
    speedup: float = 1.0
    status: str = "ok"  # "ok" | "memory_guard" | "concurrency_throttled" | "fallback_gate" | "probing" | "zero_crash_fallback"


class TreeAdaptiveController(BudgetController):
    """Hardware-aware TreeAdaptive Speculative Controller v0.2.

    Dynamically selects optimal 2D tree topology (T0, T4, T8, T16, T32) against the GPU cost model,
    enforces dual-tier memory guard with hysteresis, throttles concurrency under high load,
    and provides adaptive probing with zero-crash resilience.
    """

    def __init__(
        self,
        cost_model: Optional[FlexCostModel] = None,
        candidate_topologies: Optional[List[str]] = None,
        min_free_vram_mb: float = 1500.0,
        kv_cache_high_pct: float = 90.0,
        kv_cache_recover_pct: float = 80.0,
        cooldown_steps: int = 3,
        window_size: int = 10,
        ema_beta: float = 0.8,
        initial_alpha: float = 0.8,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            ema_beta=ema_beta,
            initial_alpha=initial_alpha,
            min_free_vram_mb=min_free_vram_mb,
            **kwargs,
        )
        self.cost_model = cost_model or FlexCostModel()
        self.candidate_topologies = candidate_topologies or ["T0", "T4", "T8", "T16", "T32"]
        self.kv_cache_high_pct = kv_cache_high_pct
        self.kv_cache_recover_pct = kv_cache_recover_pct
        self.cooldown_steps = cooldown_steps

        self._step_count: int = 0
        self._fallback_steps: int = 0
        self._memory_guard_active: bool = False
        self._guard_trigger_source: str = "none"
        self._cooldown_counter: int = 0
        self._guard_reason: str = ""
        self._last_selected_depth: int = 0
        self._last_selected_topology: str = "T0"

    def reset(self, initial_alpha: Optional[float] = None) -> None:
        """Resets controller state when beginning a new sequence/prompt."""
        super().reset(initial_alpha=initial_alpha)
        self._step_count = 0
        self._fallback_steps = 0
        self._memory_guard_active = False
        self._guard_trigger_source = "none"
        self._cooldown_counter = 0
        self._guard_reason = ""
        self._last_selected_depth = 0
        self._last_selected_topology = "T0"

    @staticmethod
    def _is_valid_vram(free_vram_mb: Optional[float]) -> bool:
        """Validates that VRAM telemetry is present, finite, non-negative, and not a sentinel value."""
        if free_vram_mb is None or math.isnan(free_vram_mb) or math.isinf(free_vram_mb):
            return False
        return 0.0 <= free_vram_mb < SENTINEL_VRAM_THRESHOLD_MB

    @staticmethod
    def _is_valid_kv(kv_cache_usage_pct: Optional[float]) -> bool:
        """Validates that KV-cache usage percentage is present, finite, and in [0, 100]."""
        if kv_cache_usage_pct is None or math.isnan(kv_cache_usage_pct) or math.isinf(kv_cache_usage_pct):
            return False
        return 0.0 <= kv_cache_usage_pct <= 100.0


    def get_effective_alpha(self) -> float:
        """Returns unbiased alpha taking into account warm-up bias correction."""
        if self._step_count == 0:
            return self.initial_alpha

        beta_power = self.ema_beta ** self._step_count
        denominator = 1.0 - beta_power
        if denominator <= 1e-6:
            return self.alpha_ema

        unbiased = (self.alpha_ema - self.initial_alpha * beta_power) / denominator
        return min(max(unbiased, 0.0), 1.0)

    def get_stats(self) -> dict:
        """Exports runtime telemetry matching Interface A v0.2 contract."""
        tree_budgets = [h["k"] for h in self._history]
        topologies = [h.get("topology_name", "T0") for h in self._history]
        return {
            "alpha_ema": self.alpha_ema,
            "effective_alpha": self.get_effective_alpha(),
            "tree_budgets": tree_budgets,
            "topologies": topologies,
            "step_count": self._step_count,
            "mean_k": (sum(tree_budgets) / len(tree_budgets)) if tree_budgets else 0.0,
            "history": list(self._history),
            "memory_guard_active": self._memory_guard_active,
            "fallback_steps": self._fallback_steps,
        }

    def update(self, num_accepted: int, num_drafted: int) -> float:
        """Updates rolling alpha telemetry from the latest step verification.

        Normalizes by active topology depth d to avoid alpha collapse on 2D trees (where N > d).
        """
        self._step_count += 1
        d = self._last_selected_depth

        if d > 0 and num_drafted > d:
            # 2D tree topology: num_drafted is total nodes N, but max accepted along a path is d
            step_alpha = min(max(num_accepted / d, 0.0), 1.0)
        elif num_drafted > 0:
            step_alpha = min(max(num_accepted / num_drafted, 0.0), 1.0)
        else:
            step_alpha = self.alpha_ema

        self._window.append(step_alpha)
        self.alpha_ema = self.ema_beta * self.alpha_ema + (1.0 - self.ema_beta) * step_alpha
        return self.alpha_ema

    def _get_probe_interval(self, batch_size: int) -> int:
        """Returns load-aware probing interval T_probe(B)."""
        if batch_size <= 4:
            return 15
        elif batch_size <= 16:
            return 20
        else:
            return 30

    def _get_throttled_candidates(self, batch_size: int) -> List[str]:
        """Filters candidate topologies according to concurrency throttling tier."""
        if batch_size <= 4:
            allowed = ["T0", "T4", "T8", "T16", "T32"]
        elif batch_size <= 12:
            allowed = ["T0", "T4", "T8", "T16"]
        elif batch_size <= 24:
            allowed = ["T0", "T4", "T8"]
        else:
            allowed = ["T0", "T4"]

        return [t for t in self.candidate_topologies if t in allowed]

    def _make_decision(
        self,
        topology_name: str,
        num_nodes: int,
        alpha_effective: float,
        alpha_star: Optional[float],
        speedup: float,
        status: str,
        reason: str,
        expected_tokens: float,
        depth: int,
    ) -> TreeAdaptiveDecision:
        self._last_selected_depth = depth
        self._last_selected_topology = topology_name
        decision = TreeAdaptiveDecision(
            k=num_nodes,
            alpha_ema=self.alpha_ema,
            expected_tokens=expected_tokens,
            reason=reason,
            topology_name=topology_name,
            num_nodes=num_nodes,
            alpha_effective=alpha_effective,
            alpha_star=alpha_star,
            speedup=speedup,
            status=status,
        )
        self._history.append({
            "k": num_nodes,
            "topology_name": topology_name,
            "alpha_effective": alpha_effective,
            "alpha_star": alpha_star,
            "speedup": speedup,
            "status": status,
            "reason": reason,
        })
        return decision

    def select_budget(
        self,
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
        kv_cache_usage_pct: float = 0.0,
    ) -> TreeAdaptiveDecision:
        """Selects optimal tree topology considering memory, load, and dynamic break-even speedup."""
        try:
            # ------------------------------------------------------------------
            # TẦNG 1: DUAL-TIER MEMORY GUARD & HYSTERESIS COOLDOWN
            # ------------------------------------------------------------------
            vram_valid = self._is_valid_vram(free_vram_mb)
            kv_valid = self._is_valid_kv(kv_cache_usage_pct)

            if not kv_valid:
                # Corrupt KV telemetry triggers defensive guard immediately
                is_vram_starved = False
                is_kv_pressured = True
                effective_kv_high = self.kv_cache_high_pct
                effective_kv_recover = self.kv_cache_recover_pct
                trigger_source = "kv"
            else:
                if vram_valid:
                    is_vram_starved = free_vram_mb < self.min_free_vram_mb
                    effective_kv_high = self.kv_cache_high_pct
                    effective_kv_recover = self.kv_cache_recover_pct
                else:
                    # Missing/sentinel VRAM: tight KV guard mode (75% / 65%)
                    is_vram_starved = False
                    effective_kv_high = 75.0
                    effective_kv_recover = 65.0

                is_kv_pressured = kv_cache_usage_pct > effective_kv_high
                if is_vram_starved and is_kv_pressured:
                    trigger_source = "both"
                elif is_vram_starved:
                    trigger_source = "vram"
                elif is_kv_pressured:
                    trigger_source = "kv"
                else:
                    trigger_source = "none"

            if is_vram_starved or is_kv_pressured:
                self._memory_guard_active = True
                self._guard_trigger_source = trigger_source
                self._cooldown_counter = self.cooldown_steps
                self._fallback_steps = 0  # Reset probe counter under pressure
                reason_parts = []
                if is_vram_starved:
                    reason_parts.append(f"free VRAM: {free_vram_mb:.0f}MB < {self.min_free_vram_mb:.0f}MB")
                if is_kv_pressured:
                    if not kv_valid:
                        reason_parts.append(f"invalid KV telemetry: {kv_cache_usage_pct}")
                    elif not vram_valid:
                        reason_parts.append(f"KV cache: {kv_cache_usage_pct:.1f}% > tight threshold {effective_kv_high:.1f}% (missing VRAM)")
                    else:
                        reason_parts.append(f"KV cache: {kv_cache_usage_pct:.1f}% > {effective_kv_high:.1f}%")
                self._guard_reason = " | ".join(reason_parts)

                return self._make_decision(
                    topology_name="T0",
                    num_nodes=0,
                    alpha_effective=self.get_effective_alpha(),
                    alpha_star=None,
                    speedup=1.0,
                    status="memory_guard",
                    reason=f"Memory guard active ({self._guard_reason})",
                    expected_tokens=1.0,
                    depth=0,
                )

            if self._memory_guard_active:
                # Recovery condition check
                if self._guard_trigger_source in ["vram", "both"]:
                    # Must have VALID VRAM reading that satisfies threshold AND safe KV
                    can_recover = (
                        vram_valid
                        and (free_vram_mb >= self.min_free_vram_mb)
                        and kv_valid
                        and (kv_cache_usage_pct <= effective_kv_recover)
                    )
                else:
                    # Pure KV trigger (or telemetry was already missing)
                    can_recover = kv_valid and (kv_cache_usage_pct <= effective_kv_recover)

                if self._cooldown_counter > 0:
                    self._cooldown_counter -= 1
                    self._fallback_steps = 0
                    return self._make_decision(
                        topology_name="T0",
                        num_nodes=0,
                        alpha_effective=self.get_effective_alpha(),
                        alpha_star=None,
                        speedup=1.0,
                        status="memory_guard",
                        reason=f"Memory guard cooldown ({self._cooldown_counter + 1} steps remaining; {self._guard_reason})",
                        expected_tokens=1.0,
                        depth=0,
                    )
                else:
                    if can_recover:
                        self._memory_guard_active = False
                        self._guard_trigger_source = "none"
                    else:
                        self._fallback_steps = 0
                        return self._make_decision(
                            topology_name="T0",
                            num_nodes=0,
                            alpha_effective=self.get_effective_alpha(),
                            alpha_star=None,
                            speedup=1.0,
                            status="memory_guard",
                            reason=f"Memory guard holding (conditions not met: VRAM={free_vram_mb}MB valid={vram_valid}, KV={kv_cache_usage_pct}%)",
                            expected_tokens=1.0,
                            depth=0,
                        )

            # ------------------------------------------------------------------
            # TẦNG 2: TÍNH TÍN HIỆU HIỆU DỤNG (UNBIASED EMA ALPHA)
            # ------------------------------------------------------------------
            alpha_effective = self.get_effective_alpha()

            # Lookup reference alpha_star for T4
            t4_result = self.cost_model.solve_alpha_star("T4", batch_size)
            ref_alpha_star_t4 = t4_result.alpha_star if t4_result.is_feasible else None

            # ------------------------------------------------------------------
            # TẦNG 3: CONCURRENCY THROTTLING (KẸP TRẦN TOPO THEO BATCH SIZE)
            # ------------------------------------------------------------------
            allowed_candidates = self._get_throttled_candidates(batch_size)

            if batch_size > 24 and alpha_effective < 0.85:
                # Concurrency hard throttle to T0 with adaptive probing cycle
                self._fallback_steps += 1
                probe_interval = self._get_probe_interval(batch_size)
                if self._fallback_steps >= probe_interval:
                    self._fallback_steps = 0
                    t4_topo = get_tree_topology("T4")
                    exp_tok = t4_topo.lookup_expected_acceptance(alpha_effective)
                    return self._make_decision(
                        topology_name="T4",
                        num_nodes=4,
                        alpha_effective=alpha_effective,
                        alpha_star=ref_alpha_star_t4,
                        speedup=1.0,
                        status="probing",
                        reason=f"Adaptive probing (interval {probe_interval} steps at B={batch_size})",
                        expected_tokens=exp_tok,
                        depth=4,
                    )
                else:
                    return self._make_decision(
                        topology_name="T0",
                        num_nodes=0,
                        alpha_effective=alpha_effective,
                        alpha_star=ref_alpha_star_t4,
                        speedup=1.0,
                        status="concurrency_throttled",
                        reason=f"Concurrency throttling at B={batch_size} > 24 with alpha={alpha_effective:.3f} < 0.85",
                        expected_tokens=1.0,
                        depth=0,
                    )

            # ------------------------------------------------------------------
            # TẦNG 4: DYNAMIC FALLBACK GATE & ARGMAX SPEEDUP
            # ------------------------------------------------------------------
            best_topo, best_speedup = self.cost_model.select_optimal_topology(
                alpha=alpha_effective,
                batch_size=batch_size,
                candidate_topologies=allowed_candidates,
            )

            if best_topo == "T0" or best_speedup <= 1.0:
                self._fallback_steps += 1
                probe_interval = self._get_probe_interval(batch_size)
                if self._fallback_steps >= probe_interval:
                    self._fallback_steps = 0
                    t4_topo = get_tree_topology("T4")
                    exp_tok = t4_topo.lookup_expected_acceptance(alpha_effective)
                    return self._make_decision(
                        topology_name="T4",
                        num_nodes=4,
                        alpha_effective=alpha_effective,
                        alpha_star=ref_alpha_star_t4,
                        speedup=1.0,
                        status="probing",
                        reason=f"Adaptive probing (interval {probe_interval} steps at B={batch_size})",
                        expected_tokens=exp_tok,
                        depth=4,
                    )
                else:
                    return self._make_decision(
                        topology_name="T0",
                        num_nodes=0,
                        alpha_effective=alpha_effective,
                        alpha_star=ref_alpha_star_t4,
                        speedup=1.0,
                        status="fallback_gate",
                        reason=f"Dynamic fallback gate (max speedup {best_speedup:.3f} <= 1.0, step {self._fallback_steps}/{probe_interval})",
                        expected_tokens=1.0,
                        depth=0,
                    )

            # Speculative tree selected
            self._fallback_steps = 0
            topo_obj = get_tree_topology(best_topo)
            num_nodes = topo_obj.num_tokens
            depth = topo_obj.depth
            exp_tok = topo_obj.lookup_expected_acceptance(alpha_effective)
            solver_res = self.cost_model.solve_alpha_star(best_topo, batch_size)
            best_alpha_star = solver_res.alpha_star if solver_res.is_feasible else None

            return self._make_decision(
                topology_name=best_topo,
                num_nodes=num_nodes,
                alpha_effective=alpha_effective,
                alpha_star=best_alpha_star,
                speedup=best_speedup,
                status="ok",
                reason=f"Optimal topology {best_topo} (Speedup={best_speedup:.3f}x, alpha*={best_alpha_star})",
                expected_tokens=exp_tok,
                depth=depth,
            )

        except Exception as exc:
            # ------------------------------------------------------------------
            # TẦNG 5: ZERO-CRASH EXCEPTION HANDLER
            # ------------------------------------------------------------------
            logger.warning("TreeAdaptiveController zero-crash triggered: %s", exc, exc_info=True)
            return self._make_decision(
                topology_name="T0",
                num_nodes=0,
                alpha_effective=self.alpha_ema,
                alpha_star=None,
                speedup=1.0,
                status="zero_crash_fallback",
                reason=f"Zero-crash fallback due to exception: {exc}",
                expected_tokens=1.0,
                depth=0,
            )
