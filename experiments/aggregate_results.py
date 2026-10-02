"""M1-02 Result Aggregation Script.

Đọc raw JSONL từ benchmark_harness.py, tính toán các chỉ số thống kê chuẩn:
- Trung vị (Median / p50) cho TTFT, TPOT, E2E
- Phân vị p95 cho TPOT
- Khoảng Min-Max và Cảnh báo phương sai: (max - min) / median > 15%
- Ghi kết quả vào experiments/results/summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SUMMARY_COLUMNS = [
    "lane",
    "workload",
    "concurrency",
    "gpu",
    "n_runs",
    "ttft_p50",
    "tpot_p50",
    "tpot_p95",
    "e2e_p50",
    "alpha",
    "tau",
    "vram_peak_mb",
    "throughput",
    "speedup_vs_vanilla",
    "variance_flag",
    "commit",
    "timestamp",
]


def calculate_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    sorted_vals = sorted(values)
    # Tính percentile 95 dùng phương pháp tuyến tính (chuẩn không cần numpy)
    idx = 0.95 * (len(sorted_vals) - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    weight = idx - lower
    val = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
    return round(val, 3)


def calculate_repeat_throughputs(records: list[dict]) -> list[float]:
    """Tính system output throughput cho từng measured repeat."""
    if not records:
        raise ValueError("Không thể tính throughput từ cell không có records.")

    repeat_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in records:
        repeat_id = record.get("repeat_id")
        run_id = record.get("run_id")

        if not repeat_id or not run_id:
            raise ValueError(
                "Measured record thiếu repeat identity: "
                f"repeat_id={repeat_id!r}, run_id={run_id!r}"
            )

        repeat_groups[(repeat_id, run_id)].append(record)

    repeat_throughputs = []

    for (repeat_id, run_id), repeat_records in repeat_groups.items():
        durations = [r.get("run_duration_s") for r in repeat_records]

        if any(duration is None for duration in durations):
            raise ValueError(
                "Repeat thiếu run_duration_s: "
                f"repeat_id={repeat_id}, run_id={run_id}"
            )

        if any(not isinstance(duration, (int, float)) for duration in durations):
            raise ValueError(
                "Repeat có run_duration_s không phải số: "
                f"repeat_id={repeat_id}, run_id={run_id}"
            )

        repeat_duration = float(durations[0])

        if repeat_duration <= 0:
            raise ValueError(
                "Repeat có run_duration_s <= 0: "
                f"repeat_id={repeat_id}, run_id={run_id}"
            )

        if any(
            not math.isclose(
                float(duration),
                repeat_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for duration in durations
        ):
            raise ValueError(
                "Các records trong cùng repeat có "
                "run_duration_s không nhất quán: "
                f"repeat_id={repeat_id}, run_id={run_id}, "
                f"durations={sorted(set(durations))}"
            )

        total_output_tokens = sum(
            r.get("output_tokens", 0)
            for r in repeat_records
            if isinstance(r.get("output_tokens"), int)
        )

        repeat_throughputs.append(
            total_output_tokens / repeat_duration
        )

    return repeat_throughputs


def aggregate_raw_jsonl(raw_path: Path) -> list[dict]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {raw_path}")

    cells = defaultdict(list)
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Chỉ lấy các record thuộc giai đoạn measured và thành công
            if record.get("phase") == "measured" and record.get("success", False):
                cell_key = (
                    record.get("lane", "vanilla"),
                    record.get("workload", "smoke"),
                    int(record.get("concurrency", 1)),
                )
                cells[cell_key].append(record)

    summaries = []
    for (lane, workload, concurrency), records in sorted(cells.items()):
        ttft_vals = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
        tpot_vals = [r["tpot_ms"] for r in records if r.get("tpot_ms") is not None]
        e2e_vals = [r["e2e_ms"] for r in records if r.get("e2e_ms") is not None]
        vram_vals = [r["run_vram_peak_mb"] for r in records if r.get("run_vram_peak_mb") is not None]

        # Đếm số lần lặp (repeats)
        unique_repeats = {r.get("repeat_id") for r in records if r.get("repeat_id")}
        n_runs = len(unique_repeats) if unique_repeats else len(records)

        # 1. Tính Median (p50)
        ttft_p50 = round(statistics.median(ttft_vals), 3) if ttft_vals else 0.0
        tpot_p50 = round(statistics.median(tpot_vals), 3) if tpot_vals else 0.0
        e2e_p50 = round(statistics.median(e2e_vals), 3) if e2e_vals else 0.0

        # 2. Tính p95
        tpot_p95 = calculate_p95(tpot_vals)

        # 3. Tính Min - Max và Cảnh báo Phương sai (docs/06: (max - min) / median > 15%)
        tpot_min = round(min(tpot_vals), 3) if tpot_vals else 0.0
        tpot_max = round(max(tpot_vals), 3) if tpot_vals else 0.0

        if tpot_p50 > 0:
            variance_ratio = (tpot_max - tpot_min) / tpot_p50
            variance_flag = variance_ratio > 0.15
        else:
            variance_ratio = 0.0
            variance_flag = False

        # Metadata
        sample = records[0]
        gpu_name = sample.get("gpu", "Tesla T4")
        commit_sha = sample.get("commit", "unknown")
        vram_peak = round(max(vram_vals), 2) if vram_vals else 0.0

        # System output throughput: tính theo repeat rồi lấy median across repeats
        repeat_throughputs = calculate_repeat_throughputs(records)
        throughput = round(statistics.median(repeat_throughputs), 2)

        summary_row = {
            "lane": lane,
            "workload": workload,
            "concurrency": concurrency,
            "gpu": gpu_name,
            "n_runs": n_runs,
            "ttft_p50": ttft_p50,
            "tpot_p50": tpot_p50,
            "tpot_p95": tpot_p95,
            "e2e_p50": e2e_p50,
            "alpha": sample.get("alpha", "N/A"),
            "tau": sample.get("tau", "N/A"),
            "vram_peak_mb": vram_peak,
            "throughput": throughput,
            "speedup_vs_vanilla": 1.0 if lane == "vanilla" else "N/A",
            "variance_flag": variance_flag,
            "commit": commit_sha,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Thêm metadata phụ để debug
            "_tpot_min": tpot_min,
            "_tpot_max": tpot_max,
            "_variance_ratio": round(variance_ratio * 100, 2),
            "_total_requests": len(records),
        }
        summaries.append(summary_row)

    return summaries


def append_to_summary_csv(summaries: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(SUMMARY_COLUMNS) + "\n")
        for row in summaries:
            line_vals = [str(row.get(col, "")) for col in SUMMARY_COLUMNS]
            f.write(",".join(line_vals) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="M1-02 Result Aggregator")
    parser.add_argument(
        "--raw",
        required=True,
        help="Đường dẫn tới file raw JSONL đo được từ benchmark_harness.py",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments/results/summary.csv",
        help="Đường dẫn file CSV tổng hợp (mặc định: experiments/results/summary.csv)",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    csv_path = Path(args.output_csv)

    print(f"\n===== ĐỐI SOÁT VÀ TỔNG HỢP SỐ LIỆU M1-02 =====")
    print(f"File raw đầu vào : {raw_path}")
    print(f"File CSV đầu ra  : {csv_path}")

    summaries = aggregate_raw_jsonl(raw_path)
    if not summaries:
        print("[CẢNH BÁO] Không tìm thấy dữ liệu measured hợp lệ trong file raw.")
        return 1

    print("\nKết quả tổng hợp chi tiết theo từng Cell:")
    print("-" * 75)
    for s in summaries:
        print(f"Cell: {s['lane']}__{s['workload']}__c{s['concurrency']}")
        print(f"  • Số request đo   : {s['_total_requests']} (qua {s['n_runs']} lần lặp / repeats)")
        print(f"  • TTFT p50 (ms)   : {s['ttft_p50']}")
        print(f"  • TPOT p50 (ms)   : {s['tpot_p50']} (Min: {s['_tpot_min']}, Max: {s['_tpot_max']})")
        print(f"  • TPOT p95 (ms)   : {s['tpot_p95']}")
        print(f"  • E2E p50 (ms)    : {s['e2e_p50']}")
        print(f"  • Độ lệch TPOT    : {s['_variance_ratio']}%")
        print(f"  • Variance Flag   : {'⚠️ BỊ NHIỄU (>15%)' if s['variance_flag'] else '✅ ỔN ĐỊNH (<=15%)'}")
        print("-" * 75)

    append_to_summary_csv(summaries, csv_path)
    print(f"\n[THÀNH CÔNG] Đã ghi {len(summaries)} dòng vào {csv_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())