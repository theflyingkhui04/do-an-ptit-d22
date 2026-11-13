"""M1-02 canonical result aggregation.

This module reads raw JSONL emitted by ``benchmark_harness.py`` and aggregates
metrics using the experiment hierarchy:

    cell -> repeat -> request

Canonical rules implemented here:
- Any measured request with ``success != True`` or
  ``metrics_complete != True`` invalidates the aggregation.
- Request-level latency metrics are summarized inside each repeat first.
- Cell-level latency metrics are the median across repeat-level summaries.
- Cell variance is computed from repeat-level TPOT p50 values.
- System output throughput is computed per repeat as:
      sum(output_tokens) / run_duration_s
  and the cell value is the median across repeats.
- Static speculative-decoding acceptance metrics are reconstructed from summed
  native vLLM counters:
      alpha = sum(accepted_draft_tokens) / sum(draft_tokens)
      tau   = 1 + sum(accepted_draft_tokens) / sum(spec_steps)
- ``speedup_vs_vanilla`` is TPOT-based and paired by identical
  ``(workload, concurrency)``:
      vanilla_tpot_p50 / lane_tpot_p50

The public ``summary.csv`` schema is intentionally kept compatible with
``docs/06-ke-hoach-thuc-nghiem.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


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


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected finite number, got {value!r}")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: NaN/Infinity is not allowed")

    return result


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected integer, got {value!r}")
    return value


def calculate_p95(values: Sequence[float]) -> float:
    """Return p95 using linear interpolation over sorted values."""
    if not values:
        raise ValueError("Cannot calculate p95 from an empty sequence.")

    if len(values) == 1:
        return round(float(values[0]), 3)

    sorted_vals = sorted(float(value) for value in values)
    idx = 0.95 * (len(sorted_vals) - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    weight = idx - lower

    value = (
        sorted_vals[lower] * (1.0 - weight)
        + sorted_vals[upper] * weight
    )
    return round(value, 3)


def _load_jsonl(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {raw_path}")

    rows: list[dict[str, Any]] = []

    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{raw_path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{raw_path}:{line_number}: expected JSON object"
                )

            rows.append(record)

    return rows


def validate_measured_records(records: Sequence[dict[str, Any]]) -> None:
    """Fail aggregation if any measured request failed or is incomplete."""
    failures: list[str] = []

    for index, record in enumerate(records):
        if record.get("phase") != "measured":
            continue

        request_id = record.get("request_id", f"row-{index}")

        if record.get("success") is not True:
            failures.append(
                f"{request_id}: success={record.get('success')!r}"
            )

        if record.get("metrics_complete") is not True:
            failures.append(
                f"{request_id}: metrics_complete="
                f"{record.get('metrics_complete')!r}"
            )

    if failures:
        preview = "; ".join(failures[:10])
        suffix = (
            f"; ... +{len(failures) - 10} more"
            if len(failures) > 10
            else ""
        )
        raise ValueError(
            "Measured request validation failed; refusing to silently "
            f"drop invalid evidence: {preview}{suffix}"
        )


def _measured_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("phase") == "measured"
    ]


def _metric_values(
    records: Sequence[dict[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []

    for record in records:
        if field not in record or record.get(field) is None:
            raise ValueError(
                f"Measured record missing required metric {field!r}: "
                f"request_id={record.get('request_id')!r}"
            )
        values.append(_finite_number(record[field], field))

    if not values:
        raise ValueError(f"No values found for metric {field!r}")

    return values


def _repeat_identity(
    record: dict[str, Any],
) -> tuple[str, str]:
    repeat_id = record.get("repeat_id")
    run_id = record.get("run_id")

    if not isinstance(repeat_id, str) or not repeat_id:
        raise ValueError(
            "Measured record missing repeat_id: "
            f"request_id={record.get('request_id')!r}"
        )

    if not isinstance(run_id, str) or not run_id:
        raise ValueError(
            "Measured record missing run_id: "
            f"request_id={record.get('request_id')!r}"
        )

    return repeat_id, run_id


def summarize_repeat(
    records: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Summarize request observations inside one measured repeat."""
    if not records:
        raise ValueError("Cannot summarize an empty repeat.")

    repeat_keys = {_repeat_identity(record) for record in records}
    if len(repeat_keys) != 1:
        raise ValueError(
            "summarize_repeat received more than one repeat identity: "
            f"{sorted(repeat_keys)!r}"
        )

    ttft_vals = _metric_values(records, "ttft_ms")
    tpot_vals = _metric_values(records, "tpot_ms")
    e2e_vals = _metric_values(records, "e2e_ms")

    durations = [
        _finite_number(record.get("run_duration_s"), "run_duration_s")
        for record in records
    ]

    repeat_duration = durations[0]

    if repeat_duration <= 0:
        raise ValueError(
            f"run_duration_s must be > 0, got {repeat_duration}"
        )

    if any(
        not math.isclose(
            duration,
            repeat_duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for duration in durations
    ):
        raise ValueError(
            "run_duration_s is inconsistent inside one repeat: "
            f"{sorted(set(durations))}"
        )

    total_output_tokens = 0
    for record in records:
        output_tokens = _required_int(
            record.get("output_tokens"),
            "output_tokens",
        )
        if output_tokens < 0:
            raise ValueError(
                f"output_tokens must be nonnegative, got {output_tokens}"
            )
        total_output_tokens += output_tokens

    vram_vals = _metric_values(records, "run_vram_peak_mb")

    return {
        "ttft_p50": round(statistics.median(ttft_vals), 3),
        "tpot_p50": round(statistics.median(tpot_vals), 3),
        "tpot_p95": calculate_p95(tpot_vals),
        "e2e_p50": round(statistics.median(e2e_vals), 3),
        "throughput": total_output_tokens / repeat_duration,
        "vram_peak_mb": max(vram_vals),
    }


def calculate_repeat_throughputs(
    records: Sequence[dict[str, Any]],
) -> list[float]:
    """Compatibility helper: system output throughput for each repeat."""
    if not records:
        raise ValueError(
            "Không thể tính throughput từ cell không có records."
        )

    repeat_groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        repeat_groups[_repeat_identity(record)].append(record)

    return [
        summarize_repeat(repeat_records)["throughput"]
        for _, repeat_records in sorted(repeat_groups.items())
    ]


def _extract_spec_metrics(
    record: dict[str, Any],
) -> dict[str, Any]:
    server_metrics = record.get("server_metrics")
    if not isinstance(server_metrics, dict):
        raise ValueError(
            "Static measured record missing server_metrics: "
            f"request_id={record.get('request_id')!r}"
        )

    spec = server_metrics.get("speculative_decoding")
    if not isinstance(spec, dict):
        raise ValueError(
            "Static measured record missing "
            "server_metrics.speculative_decoding: "
            f"request_id={record.get('request_id')!r}"
        )

    return spec


def aggregate_static_acceptance(
    records: Sequence[dict[str, Any]],
) -> tuple[float, float]:
    """Aggregate static alpha/tau from summed native vLLM counters."""
    if not records:
        raise ValueError(
            "Cannot aggregate speculative metrics from an empty cell."
        )

    accepted_total = 0
    drafted_total = 0
    steps_total = 0

    for record in records:
        spec = _extract_spec_metrics(record)

        accepted = _required_int(
            spec.get("num_accepted_draft_tokens"),
            "num_accepted_draft_tokens",
        )
        drafted = _required_int(
            spec.get("num_draft_tokens"),
            "num_draft_tokens",
        )
        steps = _required_int(
            spec.get("num_spec_steps"),
            "num_spec_steps",
        )

        if accepted < 0 or drafted < 0 or steps < 0:
            raise ValueError(
                "Speculative-decoding counters must be nonnegative."
            )

        if accepted > drafted:
            raise ValueError(
                "num_accepted_draft_tokens cannot exceed "
                "num_draft_tokens."
            )

        accepted_total += accepted
        drafted_total += drafted
        steps_total += steps

    if drafted_total <= 0:
        raise ValueError(
            "Cannot compute static alpha because summed "
            "num_draft_tokens <= 0."
        )

    if steps_total <= 0:
        raise ValueError(
            "Cannot compute static tau because summed "
            "num_spec_steps <= 0."
        )

    alpha = accepted_total / drafted_total
    tau = 1.0 + accepted_total / steps_total

    return alpha, tau


def _single_metadata_value(
    records: Sequence[dict[str, Any]],
    field: str,
    *,
    default: Any = None,
) -> Any:
    values = {
        record.get(field, default)
        for record in records
    }

    if len(values) != 1:
        raise ValueError(
            f"Cell contains inconsistent {field!r}: "
            f"{sorted(map(repr, values))}"
        )

    return next(iter(values))


def _cell_key(
    record: dict[str, Any],
) -> tuple[str, str, int]:
    lane = record.get("lane")
    workload = record.get("workload")
    concurrency = record.get("concurrency")

    if not isinstance(lane, str) or not lane:
        raise ValueError("Measured record missing valid lane.")

    if not isinstance(workload, str) or not workload:
        raise ValueError("Measured record missing valid workload.")

    if isinstance(concurrency, bool) or not isinstance(
        concurrency,
        int,
    ):
        raise ValueError(
            f"Measured record has invalid concurrency: {concurrency!r}"
        )

    if concurrency <= 0:
        raise ValueError(
            f"concurrency must be > 0, got {concurrency}"
        )

    return lane, workload, concurrency


def _aggregate_measured_records(
    measured: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not measured:
        return []

    cells: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in measured:
        cells[_cell_key(record)].append(record)

    summaries: list[dict[str, Any]] = []

    for (lane, workload, concurrency), records in sorted(
        cells.items()
    ):
        repeat_groups: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)

        for record in records:
            repeat_groups[_repeat_identity(record)].append(record)

        repeat_summaries = [
            summarize_repeat(repeat_records)
            for _, repeat_records in sorted(repeat_groups.items())
        ]

        if not repeat_summaries:
            raise ValueError(
                f"Cell {lane}/{workload}/c{concurrency} "
                "contains no repeats."
            )

        repeat_ttft = [
            summary["ttft_p50"]
            for summary in repeat_summaries
        ]
        repeat_tpot = [
            summary["tpot_p50"]
            for summary in repeat_summaries
        ]
        repeat_tpot_p95 = [
            summary["tpot_p95"]
            for summary in repeat_summaries
        ]
        repeat_e2e = [
            summary["e2e_p50"]
            for summary in repeat_summaries
        ]
        repeat_throughput = [
            summary["throughput"]
            for summary in repeat_summaries
        ]
        repeat_vram = [
            summary["vram_peak_mb"]
            for summary in repeat_summaries
        ]

        ttft_p50 = round(
            statistics.median(repeat_ttft),
            3,
        )
        tpot_p50 = round(
            statistics.median(repeat_tpot),
            3,
        )
        tpot_p95 = round(
            statistics.median(repeat_tpot_p95),
            3,
        )
        e2e_p50 = round(
            statistics.median(repeat_e2e),
            3,
        )

        tpot_min = round(min(repeat_tpot), 3)
        tpot_max = round(max(repeat_tpot), 3)

        if tpot_p50 > 0:
            variance_ratio = (
                (tpot_max - tpot_min) / tpot_p50
            )
        else:
            raise ValueError(
                f"Cell {lane}/{workload}/c{concurrency} "
                f"has non-positive TPOT p50: {tpot_p50}"
            )

        if lane in ("eagle_fixed", "eagle_adaptive"):
            # A genuine adaptive K=0 cell may have no speculation. Preserve
            # undefined alpha/tau rather than dividing by zero or inventing 0.
            specs = [_extract_spec_metrics(r) for r in records]
            for spec in specs:
                d, a, n = (spec.get(f) for f in ("num_draft_tokens", "num_accepted_draft_tokens", "num_spec_steps"))
                if not all(type(v) is int for v in (d, a, n)) or not 0 <= a <= d or n < 0:
                    raise ValueError("Invalid M4 native speculative counters")
            if sum(s["num_draft_tokens"] for s in specs):
                alpha, tau = aggregate_static_acceptance(records)
            else:
                if lane == "eagle_fixed":
                    raise ValueError("Fixed EAGLE produced no native drafts; possible silent fallback")
                alpha = tau = "N/A"
        elif lane == "static":
            alpha, tau = aggregate_static_acceptance(records)
        else:
            alpha = "N/A"
            tau = "N/A"

        gpu_name = _single_metadata_value(
            records,
            "gpu",
            default="Tesla T4",
        )
        commit_sha = _single_metadata_value(
            records,
            "commit",
            default="unknown",
        )

        experiment_id = _single_metadata_value(
            records,
            "experiment_id",
            default=None,
        )
        dataset_version = _single_metadata_value(
            records,
            "dataset_version",
            default=None,
        )
        workload_file_sha256 = _single_metadata_value(
            records,
            "workload_file_sha256",
            default=None,
        )

        summary_row: dict[str, Any] = {
            "lane": lane,
            "workload": workload,
            "concurrency": concurrency,
            "gpu": gpu_name,
            "n_runs": len(repeat_groups),
            "ttft_p50": ttft_p50,
            "tpot_p50": tpot_p50,
            "tpot_p95": tpot_p95,
            "e2e_p50": e2e_p50,
            "alpha": (
                round(alpha, 9)
                if isinstance(alpha, float)
                else alpha
            ),
            "tau": (
                round(tau, 9)
                if isinstance(tau, float)
                else tau
            ),
            "vram_peak_mb": round(max(repeat_vram), 2),
            "throughput": round(
                statistics.median(repeat_throughput),
                2,
            ),
            "speedup_vs_vanilla": (
                1.0 if lane == "vanilla" else None
            ),
            "variance_flag": variance_ratio > 0.15,
            "commit": commit_sha,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "_experiment_id": experiment_id,
            "_dataset_version": dataset_version,
            "_workload_file_sha256": workload_file_sha256,
            "_tpot_min": tpot_min,
            "_tpot_max": tpot_max,
            "_variance_ratio": round(
                variance_ratio * 100,
                2,
            ),
            "_total_requests": len(records),
        }

        summaries.append(summary_row)

    return summaries


def aggregate_raw_jsonl(raw_path: Path) -> list[dict[str, Any]]:
    """Aggregate one raw JSONL file into one or more cell summaries."""
    records = _load_jsonl(raw_path)
    validate_measured_records(records)
    measured = _measured_records(records)
    return _aggregate_measured_records(measured)


def aggregate_raw_jsonls(
    raw_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    """Aggregate multiple raw JSONL files and reject duplicate cells."""
    summaries: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, int]] = set()

    for raw_path in raw_paths:
        file_summaries = aggregate_raw_jsonl(raw_path)

        for summary in file_summaries:
            key = (
                str(summary["lane"]),
                str(summary["workload"]),
                int(summary["concurrency"]),
            )

            if key in seen_cells:
                raise ValueError(
                    "Duplicate summary cell across raw inputs: "
                    f"{key}"
                )

            seen_cells.add(key)
            summaries.append(summary)

    return sorted(
        summaries,
        key=lambda row: (
            str(row["workload"]),
            int(row["concurrency"]),
            str(row["lane"]),
        ),
    )


def apply_speedup_vs_vanilla(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply TPOT-based speedup using matched Vanilla cells.

    Pairing identity:
        workload
        concurrency

    Scientific provenance must also match when present:
        experiment_id
        Git commit
        dataset_version
        workload_file_sha256

    Formula:
        speedup_vs_vanilla =
            vanilla_tpot_p50 / lane_tpot_p50

    Rows created by older/unit-test fixtures that omit provenance
    entirely remain supported. However, if either side supplies a
    provenance field, both sides must agree exactly.
    """

    def provenance(
        row: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "experiment_id": row.get("_experiment_id"),
            "commit": row.get("commit"),
            "dataset_version": row.get("_dataset_version"),
            "workload_file_sha256": row.get(
                "_workload_file_sha256"
            ),
        }

    def validate_pair_provenance(
        vanilla_row: dict[str, Any],
        candidate_row: dict[str, Any],
        *,
        key: tuple[str, int],
    ) -> None:
        vanilla_prov = provenance(vanilla_row)
        candidate_prov = provenance(candidate_row)

        mismatches: list[str] = []

        for field in (
            "experiment_id",
            "commit",
            "dataset_version",
            "workload_file_sha256",
        ):
            vanilla_value = vanilla_prov[field]
            candidate_value = candidate_prov[field]

            # Backward-compatible unit/legacy rows may omit a
            # provenance field on both sides.
            if vanilla_value is None and candidate_value is None:
                continue

            if vanilla_value != candidate_value:
                mismatches.append(
                    f"{field}: "
                    f"vanilla={vanilla_value!r}, "
                    f"lane={candidate_value!r}"
                )

        if mismatches:
            raise ValueError(
                "Cannot compute speedup from scientifically "
                f"non-equivalent cells {key}: "
                + "; ".join(mismatches)
            )

    vanilla_by_key: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    for row in rows:
        if row.get("lane") != "vanilla":
            continue

        workload = row.get("workload")
        concurrency = row.get("concurrency")

        if not isinstance(workload, str) or not workload:
            raise ValueError(
                "Vanilla summary row missing valid workload."
            )

        if isinstance(concurrency, bool) or not isinstance(
            concurrency,
            int,
        ):
            raise ValueError(
                "Vanilla summary row has invalid concurrency: "
                f"{concurrency!r}"
            )

        key = (workload, concurrency)

        if key in vanilla_by_key:
            raise ValueError(
                f"Duplicate Vanilla baseline for {key}"
            )

        vanilla_tpot = _finite_number(
            row.get("tpot_p50"),
            "vanilla.tpot_p50",
        )

        if vanilla_tpot <= 0:
            raise ValueError(
                f"Vanilla TPOT must be > 0 for {key}, "
                f"got {vanilla_tpot}"
            )

        vanilla_by_key[key] = row

    for row in rows:
        lane = row.get("lane")
        workload = row.get("workload")
        concurrency = row.get("concurrency")

        if (
            not isinstance(workload, str)
            or isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
        ):
            raise ValueError(
                "Summary row missing valid pairing identity: "
                f"lane={lane!r}, workload={workload!r}, "
                f"concurrency={concurrency!r}"
            )

        key = (workload, concurrency)

        if lane == "vanilla":
            row["speedup_vs_vanilla"] = 1.0
            continue

        vanilla_row = vanilla_by_key.get(key)

        if vanilla_row is None:
            raise ValueError(
                "Missing matching Vanilla baseline for "
                f"lane={lane!r}, workload={workload!r}, "
                f"concurrency={concurrency}"
            )

        validate_pair_provenance(
            vanilla_row,
            row,
            key=key,
        )

        vanilla_tpot = _finite_number(
            vanilla_row.get("tpot_p50"),
            "vanilla.tpot_p50",
        )

        lane_tpot = _finite_number(
            row.get("tpot_p50"),
            f"{lane}.tpot_p50",
        )

        if lane_tpot <= 0:
            raise ValueError(
                f"{lane} TPOT must be > 0 for {key}, "
                f"got {lane_tpot}"
            )

        row["speedup_vs_vanilla"] = round(
            vanilla_tpot / lane_tpot,
            9,
        )

    return rows

def check_m2_cells(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check campaign completeness and identify noisy cells for rerun.

    This readiness check does not change aggregation, export an analysis CSV,
    or decide the empirical W3 speedup gate. All raw evidence stays intact.
    """
    expected = {
        (lane, workload, concurrency)
        for lane in ("vanilla", "static", "adaptive")
        for workload in ("W1", "W2", "W3")
        for concurrency in (1, 2, 4, 8, 16, 32)
    }
    seen = set()
    errors = []
    noisy = []
    for row in summaries:
        key = _cell_key(row)
        if key in seen:
            errors.append(f"Duplicate cell: {key}")
        seen.add(key)
        if _required_int(row.get("n_runs"), "n_runs") < 5:
            errors.append(f"{key}: requires >=5 measured repeats")
        minimum = _finite_number(row.get("_tpot_min"), "_tpot_min")
        maximum = _finite_number(row.get("_tpot_max"), "_tpot_max")
        median = _finite_number(row.get("tpot_p50"), "tpot_p50")
        if not 0 < minimum <= median <= maximum:
            errors.append(f"{key}: invalid repeat-level TPOT range")
            continue
        ratio = (maximum - minimum) / median
        if row.get("variance_flag") is not (ratio > 0.15):
            errors.append(f"{key}: variance_flag must equal variance_ratio > 0.15")
        if ratio > 0.15:
            noisy.append({"cell": key, "variance_ratio": ratio})
    if seen != expected or len(summaries) != 54:
        errors.append(
            f"Expected exactly 54 cells; missing={sorted(expected - seen)}; "
            f"unexpected={sorted(seen - expected)}"
        )
    return {
        "status": "PASS" if not errors and not noisy else "FAIL",
        "cell_count": len(summaries),
        "errors": errors,
        "noisy_cells_rerun_required": noisy,
        "variance_threshold": 0.15,
        "w3_speedup_gate": "REQUIRES_EMPIRICAL_ANALYSIS",
    }


def append_to_summary_csv(
    summaries: Sequence[dict[str, Any]],
    csv_path: Path,
) -> None:
    """Append canonical summary rows, preserving historical API behavior."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_COLUMNS,
        )

        if write_header:
            writer.writeheader()

        for row in summaries:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in SUMMARY_COLUMNS
                }
            )


def write_summary_csv(
    summaries: Sequence[dict[str, Any]],
    csv_path: Path,
) -> None:
    """Write a deterministic canonical summary from scratch."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_COLUMNS,
        )
        writer.writeheader()

        for row in summaries:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in SUMMARY_COLUMNS
                }
            )


def _print_summary(summaries: Sequence[dict[str, Any]]) -> None:
    print("\nKết quả tổng hợp chi tiết theo từng Cell:")
    print("-" * 75)

    for summary in summaries:
        print(
            "Cell: "
            f"{summary['lane']}__"
            f"{summary['workload']}__"
            f"c{summary['concurrency']}"
        )
        print(
            "  • Số request đo   : "
            f"{summary['_total_requests']} "
            f"(qua {summary['n_runs']} lần lặp / repeats)"
        )
        print(
            "  • TTFT p50 (ms)   : "
            f"{summary['ttft_p50']}"
        )
        print(
            "  • TPOT p50 (ms)   : "
            f"{summary['tpot_p50']} "
            f"(Repeat Min: {summary['_tpot_min']}, "
            f"Repeat Max: {summary['_tpot_max']})"
        )
        print(
            "  • TPOT p95 (ms)   : "
            f"{summary['tpot_p95']}"
        )
        print(
            "  • E2E p50 (ms)    : "
            f"{summary['e2e_p50']}"
        )
        print(
            "  • Độ lệch TPOT    : "
            f"{summary['_variance_ratio']}%"
        )
        print(
            "  • Variance Flag   : "
            + (
                "⚠️ BỊ NHIỄU (>15%)"
                if summary["variance_flag"]
                else "✅ ỔN ĐỊNH (<=15%)"
            )
        )
        print(
            "  • Alpha / Tau     : "
            f"{summary['alpha']} / {summary['tau']}"
        )
        print(
            "  • Throughput      : "
            f"{summary['throughput']} tok/s"
        )
        print(
            "  • Speedup vs Van. : "
            f"{summary['speedup_vs_vanilla']}"
        )
        print("-" * 75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M1-02 canonical result aggregator"
    )

    parser.add_argument(
        "--raw",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "Một hoặc nhiều raw JSONL từ benchmark_harness.py. "
            "Để tính speedup Static/Adaptive, truyền cả matching "
            "Vanilla files trong cùng invocation."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/results/summary.csv"),
        help=(
            "Đường dẫn CSV tổng hợp "
            "(mặc định: experiments/results/summary.csv)"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Ghi lại output CSV từ đầu. Khuyến nghị cho canonical "
            "summary để tránh stale/duplicate rows."
        ),
    )

    parser.add_argument(
        "--check-m2", action="store_true",
        help="Check 54 cells, >=5 repeats and >15%% noise; print JSON without writing CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check_m2:
        try:
            report = check_m2_cells(aggregate_raw_jsonls(args.raw))
        except (OSError, ValueError) as exc:
            report = {"status": "FAIL", "errors": [str(exc)]}
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1

    print(
        "\n===== ĐỐI SOÁT VÀ TỔNG HỢP SỐ LIỆU M1-02 ====="
    )

    for raw_path in args.raw:
        print(f"Raw input          : {raw_path}")

    print(f"CSV output         : {args.output_csv}")

    try:
        summaries = aggregate_raw_jsonls(args.raw)

        if not summaries:
            raise ValueError(
                "Không tìm thấy measured records để aggregate."
            )

        summaries = apply_speedup_vs_vanilla(summaries)

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1

    _print_summary(summaries)

    if args.overwrite:
        write_summary_csv(
            summaries,
            args.output_csv,
        )
    else:
        append_to_summary_csv(
            summaries,
            args.output_csv,
        )

    print(
        f"\n[THÀNH CÔNG] Đã ghi {len(summaries)} dòng "
        f"vào {args.output_csv}\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
