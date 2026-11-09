"""Adaptive Speculative Proposer Adapter for vLLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from src.proposer.controller import (
    BudgetController,
    AdaptiveDraftController,
    BudgetDecision,
    CostProfile,
)
from src.proposer.tree_adaptive import (
    TreeAdaptiveController,
    TreeAdaptiveDecision,
)


@dataclass
class ProposalOutput:
    """Output bundle from adapter proposal step."""
    draft_tokens: List[int]
    budget_k: int
    skipped_draft: bool
    reason: str
    topology_name: Optional[str] = None



class AdaptiveProposerAdapter:
    """Hooks into vLLM speculative proposal loop and manages per-request BudgetControllers."""

    def __init__(
        self,
        controller_kwargs: Optional[dict] = None,
        controller_cls: type[BudgetController] = AdaptiveDraftController,
    ):
        self.controller_kwargs = controller_kwargs or {}
        self.controller_cls = controller_cls
        self._active_sessions: Dict[str, BudgetController] = {}
        self._accepted_history: Dict[str, List[int]] = {}

    @property
    def active_request_count(self) -> int:
        return len(self._active_sessions)

    def start_request(
        self,
        request_id: str,
        initial_alpha: Optional[float] = None,
    ) -> BudgetController:
        """Initializes a new BudgetController session for a given request."""
        kwargs = dict(self.controller_kwargs)
        if initial_alpha is not None:
            kwargs["initial_alpha"] = initial_alpha
        controller = self.controller_cls(**kwargs)
        self._active_sessions[request_id] = controller
        self._accepted_history[request_id] = []
        return controller

    def get_controller(self, request_id: str) -> BudgetController:
        """Gets active controller for request, or lazily initializes one."""
        if request_id not in self._active_sessions:
            return self.start_request(request_id)
        return self._active_sessions[request_id]

    def propose(
        self,
        request_id: str,
        draft_generator: Callable[[int], List[int]],
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
        kv_cache_usage_pct: float = 0.0,
    ) -> ProposalOutput:
        """Executes one adaptive speculative proposal step.
        
        Queries BudgetController for optimal k*.
        If k* == 0 (Vanilla Fallback), completely skips draft_generator to save GPU compute.
        If k* > 0, executes draft_generator(k*) and packages the tokens.
        """
        controller = self.get_controller(request_id)
        decision: BudgetDecision = controller.select_budget(
            batch_size=batch_size,
            free_vram_mb=free_vram_mb,
            kv_cache_usage_pct=kv_cache_usage_pct,
        )

        topo = getattr(decision, "topology_name", None)

        if decision.k == 0:
            return ProposalOutput(
                draft_tokens=[],
                budget_k=0,
                skipped_draft=True,
                reason=decision.reason,
                topology_name=topo,
            )

        draft_tokens = draft_generator(decision.k)
        return ProposalOutput(
            draft_tokens=draft_tokens,
            budget_k=decision.k,
            skipped_draft=False,
            reason=decision.reason,
            topology_name=topo,
        )

    def on_verification_completed(
        self,
        request_id: str,
        num_accepted: int,
        num_drafted: int,
    ) -> float:
        """Receives verification telemetry from Target model and updates rolling alpha."""
        controller = self.get_controller(request_id)
        if request_id not in self._accepted_history:
            self._accepted_history[request_id] = []
        if num_drafted > 0:
            self._accepted_history[request_id].append(num_accepted)
        return controller.update(num_accepted=num_accepted, num_drafted=num_drafted)

    def get_telemetry(self, request_id: str) -> dict:
        """Extracts runtime telemetry snapshot matching Interface A contract."""
        controller = self.get_controller(request_id)
        stats = controller.get_stats()
        accepted_records = self._accepted_history.get(request_id, [])
        mean_accept_len = (
            float(sum(accepted_records) / len(accepted_records))
            if accepted_records
            else 0.0
        )
        telemetry = {
            "request_id": request_id,
            "lane": "adaptive",
            "alpha_ema": stats["alpha_ema"],
            "tree_budgets": stats["tree_budgets"],
            "mean_accept_len": mean_accept_len,
            "mean_k": stats.get("mean_k", 0.0),
            "step_count": stats["step_count"],
        }
        if "effective_alpha" in stats:
            telemetry["effective_alpha"] = stats["effective_alpha"]
        if "is_warmed_up" in stats:
            telemetry["is_warmed_up"] = stats["is_warmed_up"]
        if "topologies" in stats:
            telemetry["topologies"] = stats["topologies"]
        return telemetry

    def end_request(self, request_id: str) -> dict:
        """Finalizes request, returns full Interface A telemetry, and releases session memory."""
        telemetry = self.get_telemetry(request_id)
        if request_id in self._active_sessions:
            del self._active_sessions[request_id]
        if request_id in self._accepted_history:
            del self._accepted_history[request_id]
        return telemetry


