"""Live telemetry monitor for inspecting speculative decoding traces in real-time."""
from __future__ import annotations

import json
import os
import sys

try:
    from flexsds.cli.output import print_output as print, write_output
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print, write_output
import time
from typing import Any, Dict, List, Optional


class TelemetryMonitor:
    """Monitors and tails a JSONL trace file containing speculative decoding steps."""

    def __init__(self, trace_path: str = "telemetry/serving_trace.jsonl") -> None:
        self.trace_path = trace_path
        self._file_pos = 0

    def read_new_records(self) -> List[Dict[str, Any]]:
        """Reads newly appended records since the last check."""
        if not os.path.exists(self.trace_path):
            return []

        try:
            with open(self.trace_path, "r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                lines = f.readlines()
                self._file_pos = f.tell()
        except OSError:
            return []

        records: List[Dict[str, Any]] = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                if isinstance(data, dict):
                    records.append(data)
            except json.JSONDecodeError:
                # Bỏ qua dòng json chưa hoàn chỉnh hoặc hỏng
                continue
        return records

    # Per-step planning records in serving_steps.jsonl (two per request per step): not displayed.
    HIDDEN_RECORD_TYPES = frozenset({"planned_decision", "scheduled_budget"})

    @staticmethod
    def format_record(record: Dict[str, Any]) -> Optional[str]:
        """Formats one telemetry record as a row, or returns None for records not worth showing.

        Understands both files: serving_trace.jsonl (one summary per finished request) and
        serving_steps.jsonl (per-step verification feedback).
        """
        lane = record.get("lane", "adaptive")
        record_type = record.get("record_type")
        if record_type in TelemetryMonitor.HIDDEN_RECORD_TYPES:
            return None
        if record_type == "verification_feedback_skipped":
            return (f"[\033[36mStep -\033[0m] \033[33mfeedback skipped "
                    f"({record.get('lineage_status', 'unknown')})\033[0m | lane={lane}")
        # Per-request summaries (record_type "request_aggregate"; step fields are null).
        if record_type == "request_aggregate" or (record_type is None and "step_count" in record):
            def num(key: str, digits: int) -> str:
                value = record.get(key)
                return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"
            request = str(record.get("request_id", "-"))[:26]
            return (f"[\033[36mRequest {request}\033[0m] steps={record.get('step_count', '-')} | "
                    f"mean_k={num('mean_k', 2)} | accept_len={num('mean_accept_len', 2)} | "
                    f"alpha={num('alpha_ema', 2)} | lane={lane}")

        # serving_steps.jsonl verification rows carry the scheduler step of their drafts.
        step = record.get("step", record.get("source_schedule_sequence"))
        executed = record.get("executed_draft_token_count", 0)
        accepted = record.get("accepted_draft_token_count", 0)
        alpha_ema = record.get("alpha_ema", 0.0)
        k = record.get("source_batch_scalar_k", executed)

        # Tính tỷ lệ chấp nhận trực quan
        ratio_str = f"{alpha_ema:.2f}" if isinstance(alpha_ema, (int, float)) else str(alpha_ema)
        color = "\033[32m" if accepted >= 3 else ("\033[33m" if accepted >= 1 else "\033[31m")

        return (
            (f"[\033[36mStep {step}\033[0m] " if step is not None else "")
            + f"k={k} | "
            f"{color}accepted={accepted}/{executed}\033[0m | "
            f"ratio={ratio_str} | "
            f"lane={lane}"
        )


def run_live_monitor(
    trace_file: str = "telemetry/serving_trace.jsonl",
    refresh_rate: float = 1.0,
) -> int:
    """CLI entry point for streaming telemetry trace updates."""
    monitor = TelemetryMonitor(trace_path=trace_file)

    print("\n\033[1;37mFlexSDS Live Telemetry Monitor\033[0m")
    print(f"  Trace file: \033[36m{trace_file}\033[0m")
    print(f"  Refresh rate: \033[33m{refresh_rate}s\033[0m | \033[90mPress Ctrl+C to stop\033[0m\n")

    if not os.path.exists(trace_file):
        print(f"  \033[90m(Waiting for server to create trace file at {trace_file}...)\033[0m")

    try:
        while True:
            records = monitor.read_new_records()
            for r in records:
                line = monitor.format_record(r)
                if line is not None:
                    print(f"  {line}")
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        print("\n\033[90mTelemetry monitor stopped.\033[0m")
        return 0
