"""Adaptive Speculative Proposer Adapter for vLLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from src.proposer.controller import BudgetController, BudgetDecision, CostProfile


@dataclass
class ProposalOutput:
    """Output bundle from adapter proposal step."""
    draft_tokens: List[int]
    budget_k: int
    skipped_draft: bool
    reason: str


class AdaptiveProposerAdapter:
    """Hooks into vLLM speculative proposal loop and manages per-request BudgetControllers."""

    def __init__(self, controller_kwargs: Optional[dict] = None):
        self.controller_kwargs = controller_kwargs or {}
        self._active_sessions: Dict[str, BudgetController] = {}

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
        controller = BudgetController(**kwargs)
        self._active_sessions[request_id] = controller
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
        )

        if decision.k == 0:
            return ProposalOutput(
                draft_tokens=[],
                budget_k=0,
                skipped_draft=True,
                reason=decision.reason,
            )

        draft_tokens = draft_generator(decision.k)
        return ProposalOutput(
            draft_tokens=draft_tokens,
            budget_k=decision.k,
            skipped_draft=False,
            reason=decision.reason,
        )

