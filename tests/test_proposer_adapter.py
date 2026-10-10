"""Unit tests for AdaptiveProposerAdapter and ProposalOutput."""
import pytest
from src.proposer.adapter import AdaptiveProposerAdapter, ProposalOutput

def test_adapter_initialization():
    adapter = AdaptiveProposerAdapter()
    assert adapter.active_request_count == 0

def test_proposal_output_structure():
    output = ProposalOutput(
        draft_tokens=[10, 20, 30],
        budget_k=3,
        skipped_draft=False,
        reason="Optimal budget k=3",
    )
    assert output.budget_k == 3
    assert len(output.draft_tokens) == 3
    assert not output.skipped_draft
