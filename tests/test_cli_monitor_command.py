"""Tests for live telemetry monitor reading serving traces."""
import json
import pytest
from unittest.mock import MagicMock
from flexsds.client.monitor import TelemetryMonitor, run_live_monitor


def test_monitor_waits_for_telemetry_file(tmp_path):
    trace_path = tmp_path / "serving_trace.jsonl"
    monitor = TelemetryMonitor(str(trace_path))
    # File chưa tồn tại -> không crash, trả về danh sách rỗng
    records = monitor.read_new_records()
    assert len(records) == 0


def test_monitor_parses_records_correctly(tmp_path):
    trace_path = tmp_path / "serving_trace.jsonl"
    record1 = {
        "step": 1,
        "lane": "adaptive",
        "executed_draft_token_count": 4,
        "accepted_draft_token_count": 3,
        "alpha_ema": 0.75,
        "source_batch_scalar_k": 4,
    }
    record2 = {
        "step": 2,
        "lane": "adaptive",
        "executed_draft_token_count": 5,
        "accepted_draft_token_count": 4,
        "alpha_ema": 0.80,
        "source_batch_scalar_k": 5,
    }

    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record1) + "\n")

    monitor = TelemetryMonitor(str(trace_path))
    records = monitor.read_new_records()
    assert len(records) == 1
    assert records[0]["accepted_draft_token_count"] == 3
    assert records[0]["alpha_ema"] == 0.75

    # Ghi tiếp record 2
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record2) + "\n")

    records2 = monitor.read_new_records()
    assert len(records2) == 1
    assert records2[0]["accepted_draft_token_count"] == 4
    assert records2[0]["source_batch_scalar_k"] == 5


def test_monitor_handles_malformed_json_lines(tmp_path):
    trace_path = tmp_path / "serving_trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("{malformed json line\n")
        f.write(json.dumps({"step": 1, "lane": "scalar_k", "accepted_draft_token_count": 2}) + "\n")

    monitor = TelemetryMonitor(str(trace_path))
    records = monitor.read_new_records()
    # Bỏ qua dòng lỗi, parse được dòng hợp lệ
    assert len(records) == 1
    assert records[0]["step"] == 1


def test_monitor_format_record():
    record = {
        "step": 10,
        "lane": "adaptive",
        "executed_draft_token_count": 4,
        "accepted_draft_token_count": 3,
        "alpha_ema": 0.75,
        "source_batch_scalar_k": 4,
    }
    line = TelemetryMonitor.format_record(record)
    assert "Step 10" in line
    assert "k=4" in line
    assert "accepted=3/4" in line or "chấp nhận=3/4" in line
    assert "0.75" in line


def test_main_monitor_dispatch(monkeypatch):
    from flexsds.cli.main import main
    called = {}

    def mock_run_monitor(trace_file, refresh_rate):
        called["trace_file"] = trace_file
        called["refresh_rate"] = refresh_rate
        return 0

    monkeypatch.setattr("flexsds.cli.main.run_live_monitor", mock_run_monitor)
    code = main(["monitor", "--trace-file", "custom_trace.jsonl", "--refresh-rate", "0.5"])
    assert code == 0
    assert called["trace_file"] == "custom_trace.jsonl"
    assert called["refresh_rate"] == 0.5


def test_monitor_formats_the_per_request_trace_record():
    # telemetry/serving_trace.jsonl (the default file) holds one summary per finished request.
    # Shape of src/server/budget_planner.py request telemetry (record_type="request_aggregate").
    record = {"record_type": "request_aggregate", "decision_scope": "last_planned_decision",
              "last_planned_decision": {"record_type": "planned_decision", "batch_k_selected": 2},
              "request_id": "chatcmpl-9703297d6964b8f2-b77a0b98", "lane": "adaptive", "alpha_ema": 0.4579,
              "mean_accept_len": 1.0345, "mean_k": 2.8065, "step_count": 29, "geometric_alpha_mle": 0.6485,
              # real serving_trace.jsonl rows carry these step fields as null
              "executed_draft_token_count": None, "tree_execution_verified": False}
    line = TelemetryMonitor.format_record(record)
    assert "Request chatcmpl-9703297d" in line
    assert "steps=29" in line and "mean_k=2.81" in line and "accept_len=1.03" in line
    assert "alpha=0.46" in line and "lane=adaptive" in line
    assert "None" not in line


def test_monitor_shows_skipped_feedback_and_hides_planning_records():
    skipped = {"record_type": "verification_feedback_skipped", "request_id": "chatcmpl-8e38a0b97c790850-x",
               "lane": "adaptive", "lineage_status": "stale-or-mismatched-lineage"}
    assert "feedback skipped (stale-or-mismatched-lineage)" in TelemetryMonitor.format_record(skipped)
    for record_type in ("planned_decision", "scheduled_budget"):
        assert TelemetryMonitor.format_record({"record_type": record_type, "lane": "adaptive"}) is None


def test_run_live_monitor_prints_only_displayable_records(tmp_path, monkeypatch, capsys):
    from flexsds.client import monitor as mod

    trace = tmp_path / "steps.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in (
        {"record_type": "planned_decision", "lane": "adaptive"},
        {"record_type": "verification_feedback", "lane": "adaptive", "executed_draft_token_count": 1,
         "accepted_draft_token_count": 1, "alpha_ema": 0.84, "source_batch_scalar_k": 2},
    )) + "\n", encoding="utf-8")

    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(mod.time, "sleep", stop)
    assert mod.run_live_monitor(str(trace), refresh_rate=1) == 0
    out = capsys.readouterr().out
    assert "k=2" in out and "accepted=1/1" in out
    assert out.count("accepted=") == 1


def test_monitor_step_column_uses_the_schedule_sequence():
    record = {"record_type": "verification_feedback", "lane": "adaptive", "executed_draft_token_count": 1,
              "accepted_draft_token_count": 1, "alpha_ema": 0.84, "source_batch_scalar_k": 2,
              "source_schedule_sequence": 132564}
    assert "Step 132564" in TelemetryMonitor.format_record(record)
    bare = {"lane": "adaptive", "executed_draft_token_count": 2, "accepted_draft_token_count": 1, "alpha_ema": 0.5}
    line = TelemetryMonitor.format_record(bare)
    assert "Step -" not in line and "accepted=1/2" in line
