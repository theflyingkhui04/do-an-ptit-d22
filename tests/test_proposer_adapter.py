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


def test_propose_with_active_draft():
    adapter = AdaptiveProposerAdapter()
    req_id = "req_001"

    # Mock draft generator sinh đúng k tokens
    mock_draft_fn = lambda k: [100 + i for i in range(k)]

    # Mặc định initial_alpha=0.8, k* in {3, 4}
    output = adapter.propose(request_id=req_id, draft_generator=mock_draft_fn, batch_size=1)
    assert output.budget_k in [3, 4]
    assert len(output.draft_tokens) == output.budget_k
    assert not output.skipped_draft
    assert adapter.active_request_count == 1


def test_propose_fallback_skips_draft_generator():
    adapter = AdaptiveProposerAdapter(
        controller_kwargs={"initial_alpha": 0.40, "fallback_threshold": 0.65}
    )
    req_id = "req_fallback"

    called = False
    def mock_draft_fn(k):
        nonlocal called
        called = True
        return [999] * k

    output = adapter.propose(request_id=req_id, draft_generator=mock_draft_fn)
    assert output.budget_k == 0
    assert output.draft_tokens == []
    assert output.skipped_draft
    # Hàm draft_generator không được phép chạy khi k=0 để tiết kiệm GPU compute
    assert not called

