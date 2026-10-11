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


def test_simulation_decoding_loop():
    adapter = AdaptiveProposerAdapter(controller_kwargs={"fallback_threshold": 0.65})
    req_id = "req_sim_loop"
    adapter.start_request(req_id, initial_alpha=0.9)

    # Bước 1..3: alpha cao (3/3 đúng) -> giữ k=3 hoặc 4
    for _ in range(3):
        out = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
        assert out.budget_k > 0
        adapter.on_verification_completed(req_id, num_accepted=out.budget_k, num_drafted=out.budget_k)

    # Bước 4..6: alpha tụt dốc (0/3 đúng như W3) -> adapter nhận diện và chuyển về k=0
    for _ in range(4):
        out = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
        if not out.skipped_draft:
            adapter.on_verification_completed(req_id, num_accepted=0, num_drafted=out.budget_k)

    # Bước 7: Phải rơi vào Vanilla Fallback (k=0)
    out_last = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
    assert out_last.budget_k == 0
    assert out_last.skipped_draft

    # Kết thúc request: xuất telemetry và giải phóng bộ nhớ
    telemetry = adapter.end_request(req_id)
    assert "tree_budgets" in telemetry
    assert 0 in telemetry["tree_budgets"]  # Đã có bước k=0
    assert telemetry["tree_budgets"][0] > 0  # Bước đầu tiên có nháp
    assert adapter.active_request_count == 0  # Đã giải phóng RAM


def test_adapter_memory_cleanup():
    adapter = AdaptiveProposerAdapter()
    for i in range(10):
        req_id = f"req_{i}"
        adapter.propose(req_id, draft_generator=lambda k: [1] * k)
    assert adapter.active_request_count == 10

    for i in range(10):
        adapter.end_request(f"req_{i}")
    assert adapter.active_request_count == 0


def test_adapter_with_adaptive_draft_controller():
    from src.proposer.controller import AdaptiveDraftController

    adapter = AdaptiveProposerAdapter()
    req_id = "req_adaptive"
    controller = adapter.get_controller(req_id)
    assert isinstance(controller, AdaptiveDraftController)
    assert controller.window_size == 10
    assert controller.get_window_samples() == []

    # Giả lập 2 bước decode
    out1 = adapter.propose(req_id, draft_generator=lambda k: [42] * k)
    adapter.on_verification_completed(req_id, num_accepted=2, num_drafted=out1.budget_k)

    stats = adapter.end_request(req_id)
    assert "effective_alpha" in stats
    assert "is_warmed_up" in stats
    assert stats["is_warmed_up"] is False  # 1 bước < 10 bước
    assert stats["mean_accept_len"] == 2.0  # 2 tokens accepted
    assert stats["mean_k"] == float(out1.budget_k)


def test_adapter_telemetry_semantics_mean_accept_len_vs_mean_k():
    adapter = AdaptiveProposerAdapter()
    req_id = "req_telemetry_check"

    # Step 1: propose draft, target accepts 2
    out1 = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
    adapter.on_verification_completed(req_id, num_accepted=2, num_drafted=out1.budget_k)

    # Step 2: propose draft, target accepts 1
    out2 = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
    adapter.on_verification_completed(req_id, num_accepted=1, num_drafted=out2.budget_k)

    telemetry = adapter.end_request(req_id)
    # mean_accept_len is mean of [2, 1] = 1.5
    assert telemetry["mean_accept_len"] == 1.5
    # mean_k is mean of proposed budgets
    assert telemetry["mean_k"] == (out1.budget_k + out2.budget_k) / 2.0
    assert telemetry["mean_accept_len"] != telemetry["mean_k"]
    # Session memory cleaned up
    assert req_id not in adapter._accepted_history
    assert req_id not in adapter._active_sessions



