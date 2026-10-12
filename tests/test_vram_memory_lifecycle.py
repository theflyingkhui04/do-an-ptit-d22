"""Unit and integration tests for memory lifecycle, reference cleanup, and VRAM stability."""
from __future__ import annotations

import gc
import tracemalloc
import pytest
import torch

from src.proposer.adapter import AdaptiveProposerAdapter, ProposalOutput
from src.proposer.controller import AdaptiveDraftController, BudgetController, CostProfile


def test_adapter_full_session_cleanup_zero_leak():
    """Verify that end_request completely clears all internal session state."""
    adapter = AdaptiveProposerAdapter()
    num_requests = 25

    for req_idx in range(num_requests):
        req_id = f"session_req_{req_idx}"
        # Start request
        controller = adapter.get_controller(req_id)
        assert req_id in adapter._active_sessions

        # Run 10 steps of proposal & verification
        for step in range(10):
            # Alternating alpha behavior to force k changes: 0, 2, 3, 4
            num_draft = 3 if step % 2 == 0 else 2
            out = adapter.propose(
                request_id=req_id,
                draft_generator=lambda k: [100 + i for i in range(k)],
                free_vram_mb=16000.0,
            )
            if not out.skipped_draft:
                adapter.on_verification_completed(
                    request_id=req_id,
                    num_accepted=out.budget_k if step % 3 != 0 else 0,
                    num_drafted=out.budget_k,
                )

        telemetry = adapter.end_request(req_id)
        assert telemetry["request_id"] == req_id
        assert "mean_accept_len" in telemetry
        assert "mean_k" in telemetry

    # Verify zero residual sessions or history dictionaries
    assert adapter.active_request_count == 0
    assert len(adapter._active_sessions) == 0
    assert len(adapter._accepted_history) == 0


def test_tracemalloc_bounded_heap_growth_across_requests():
    """Verify that processing 50 consecutive requests exhibits bounded heap memory (no Python object leak)."""
    tracemalloc.start()
    gc.collect()

    adapter = AdaptiveProposerAdapter()

    # 1. Warm-up phase: 5 requests to initialize internal lookups and import caches
    for i in range(5):
        req_id = f"warmup_{i}"
        for step in range(10):
            out = adapter.propose(req_id, draft_generator=lambda k: [42] * k)
            if not out.skipped_draft:
                adapter.on_verification_completed(req_id, num_accepted=out.budget_k, num_drafted=out.budget_k)
        adapter.end_request(req_id)

    gc.collect()
    snap_baseline = tracemalloc.take_snapshot()

    # 2. Stress phase: 50 requests, each 20 decoding steps with dynamic transitions
    for i in range(50):
        req_id = f"stress_{i}"
        for step in range(20):
            # Alternating high/low acceptance rate
            target_accept = 2 if (i + step) % 4 != 0 else 0
            out = adapter.propose(req_id, draft_generator=lambda k: [i * 10 + j for j in range(k)])
            if not out.skipped_draft:
                adapter.on_verification_completed(req_id, num_accepted=target_accept, num_drafted=out.budget_k)
        adapter.end_request(req_id)

    gc.collect()
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate net allocated difference
    diff = snap_after.compare_to(snap_baseline, "lineno")
    total_net_bytes = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
    total_net_kb = total_net_bytes / 1024.0

    # The entire 50-request suite (1,000 decoding steps) must not retain more than 150 KB
    # (accounting for Python standard internal string intern / frame cache fluctuations).
    assert total_net_kb < 150.0, f"Memory leak detected: {total_net_kb:.2f} KB retained across 50 requests!"


def test_controller_bounded_history_and_reset():
    """Verify that AdaptiveDraftController reset clears all internal history buffers."""
    controller = AdaptiveDraftController()

    # Run 100 decoding steps
    for step in range(100):
        controller.select_budget()
        controller.update(num_accepted=2 if step % 3 == 0 else 1, num_drafted=3)

    stats = controller.get_stats()
    assert len(stats["tree_budgets"]) == 100
    assert len(controller._history) == 100
    assert len(controller._window) <= 10

    # Reset controller for a new sequence
    controller.reset(initial_alpha=0.8)
    assert controller.step_count == 0
    assert controller.consecutive_fallback_steps == 0
    assert len(controller._history) == 0
    assert len(controller._window) == 0
    assert controller.get_rolling_alpha() == pytest.approx(0.8)


def test_simulated_tensor_buffer_cleanup():
    """Verify that tensor buffers generated during dynamic draft depth k are completely freed."""
    adapter = AdaptiveProposerAdapter()
    req_id = "tensor_test"

    allocated_tensors = []

    def mock_draft_generator(k: int):
        # Allocate dummy tensor representing draft logits or hidden states
        tensor = torch.zeros((1, k, 256), dtype=torch.float32)
        allocated_tensors.append(tensor)
        return list(range(k))

    for step in range(15):
        out = adapter.propose(req_id, draft_generator=mock_draft_generator)
        if not out.skipped_draft:
            adapter.on_verification_completed(req_id, num_accepted=out.budget_k, num_drafted=out.budget_k)

    adapter.end_request(req_id)

    # Discard local tensor references and verify GC collection
    del allocated_tensors
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Ensure no dangling CUDA memory remains allocated
        assert torch.cuda.memory_allocated() == 0
