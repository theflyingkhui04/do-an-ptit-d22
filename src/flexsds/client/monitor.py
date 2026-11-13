"""Live telemetry monitor for inspecting speculative decoding traces in real-time."""
from __future__ import annotations

import json
import os
import sys
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

    @staticmethod
    def format_record(record: Dict[str, Any]) -> str:
        """Formats a single telemetry record into a colorful human-readable row."""
        step = record.get("step", "-")
        lane = record.get("lane", "adaptive")
        executed = record.get("executed_draft_token_count", 0)
        accepted = record.get("accepted_draft_token_count", 0)
        alpha_ema = record.get("alpha_ema", 0.0)
        k = record.get("source_batch_scalar_k", executed)

        # Tính tỷ lệ chấp nhận trực quan
        ratio_str = f"{alpha_ema:.2f}" if isinstance(alpha_ema, (int, float)) else str(alpha_ema)
        color = "\033[32m" if accepted >= 3 else ("\033[33m" if accepted >= 1 else "\033[31m")

        return (
            f"[\033[36mStep {step}\033[0m] "
            f"k={k} | "
            f"{color}chấp nhận={accepted}/{executed}\033[0m | "
            f"tỷ lệ={ratio_str} | "
            f"làn={lane}"
        )


def run_live_monitor(
    trace_file: str = "telemetry/serving_trace.jsonl",
    refresh_rate: float = 1.0,
) -> int:
    """CLI entry point for streaming telemetry trace updates."""
    monitor = TelemetryMonitor(trace_path=trace_file)

    print("\n\033[1;37mBộ theo dõi thích ứng thời gian thực (FlexSDS Live Telemetry)\033[0m")
    print(f"  Tệp theo dõi: \033[36m{trace_file}\033[0m")
    print(f"  Tần suất cập nhật: \033[33m{refresh_rate}s\033[0m | \033[90mNhấn Ctrl+C để dừng\033[0m\n")

    if not os.path.exists(trace_file):
        print(f"  \033[90m(Đang chờ máy chủ tạo tệp trace tại {trace_file}...)\033[0m")

    try:
        while True:
            records = monitor.read_new_records()
            for r in records:
                print(f"  {monitor.format_record(r)}")
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        print("\n\033[90mĐã dừng bộ theo dõi thích ứng.\033[0m")
        return 0
