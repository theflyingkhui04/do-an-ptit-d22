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
