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
