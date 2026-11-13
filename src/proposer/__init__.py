from src.proposer.controller import (
    BudgetController,
    AdaptiveDraftController,
    CostProfile,
    BudgetDecision,
)
from src.proposer.adapter import AdaptiveProposerAdapter, ProposalOutput
from src.proposer.cost_model import HardwareCostModel


def __getattr__(name):
    # Compatibility exports for explicit future-tree users. Canonical scalar
    # serving imports no tree implementation or topology catalog.
    if name in {"TreeAdaptiveController", "TreeAdaptiveDecision"}:
        from src.proposer import tree_adaptive
        return getattr(tree_adaptive, name)
    raise AttributeError(name)

__all__ = [
    "BudgetController",
    "AdaptiveDraftController",
    "TreeAdaptiveController",
    "TreeAdaptiveDecision",
    "CostProfile",
    "BudgetDecision",
    "AdaptiveProposerAdapter",
    "ProposalOutput",
    "HardwareCostModel",
]


