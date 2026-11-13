"""CPU-only 1,000 scheduler/controller lifecycle steps, not a GPU leak test."""
import json

from test_adaptive_scheduler_glue import (scheduler_module, build, make_request,
                                         add_running, retire)
from src.server.lineage import release_ledger


def test_1000_steps_with_feedback_cleanup_and_lineage_release(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path,
                  ADAPTIVE_CONTROLLER_KWARGS='{"initial_alpha":0.85,"history_limit":16}',
                  ADAPTIVE_STEP_TELEMETRY_PATH=str(tmp_path / "steps.jsonl"))
    states = set()
    for request_number in range(50):
        # Reuse the external ID: each finished session must start from fresh state.
        request = make_request("reused-id")
        add_running(sched, request)
        for step in range(20):
            sched.kv_usage = .95 if step == 5 else .1
            output = sched.schedule()
            states.add(output.num_spec_tokens_to_schedule)
            drafted = len(output.scheduled_spec_decode_tokens.get(request.request_id, []))
            retire(sched, output, [(request.request_id, drafted, drafted if step < 10 else 0)] if drafted else [])
            controller = sched._planner.adapter.get_controller(request.request_id)
            assert len(controller._history) <= 16
            assert len(controller._window) <= 10
            assert len(sched._planner.adapter._accepted_history[request.request_id]) <= 16
            assert len(sched._lineage.active) <= 1
        sched._free_request(request)
        assert sched._planner.adapter.active_request_count == 0
        assert not sched._planner.adapter._accepted_history
        assert not sched._planner._known_requests
        assert not sched._planner._last_decisions
        assert not sched._lineage.active and not sched._lineage.executions
        assert sched._lineage.generation is None
    assert {0, 3, 4} <= states
    assert len((tmp_path / "lane3.jsonl").read_text().splitlines()) == 50
    release_ledger(sched._lineage_registration)
    assert not sched._lineage.active and not sched._lineage.executions
    assert any(json.loads(line)["record_type"] == "verification_feedback"
               for line in (tmp_path / "steps.jsonl").read_text().splitlines())
