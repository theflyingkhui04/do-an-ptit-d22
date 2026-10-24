"""M2-04 three-lane analysis: canonical raw cells -> m2_summary.csv + W3 gate.

Input is the selected evidence of one canonical campaign, e.g.
``experiments/runs/m2_04/m2-04-6424c6a-t4-canonical-v1/`` on branch
``evidence/m2-04-6424c6a``::

    <campaign-root>/
    ├── cells/<lane>__<workload>__c<concurrency>/raw.jsonl
    └── triplets/<workload>__c<concurrency>.json   # COMPLETE_VALID markers

Rules (reuse ``aggregate_results.py`` unchanged):
- Every raw file must match the SHA256 recorded in its triplet marker. A
  Windows checkout with CRLF conversion changes the bytes and is rejected;
  extract blobs byte-exact (``git cat-file blob``) before analysing.
- Cell metrics follow M1-02: request -> repeat -> median across repeats.
- Static and Adaptive alpha/tau come from summed native vLLM counters
  (``server_metrics.speculative_decoding``), never from controller EMA state.
- ``speedup_vs_vanilla = vanilla_tpot_p50 / lane_tpot_p50`` with the Vanilla
  cell of the same workload, concurrency and campaign.
- W3 gate PASS iff Adaptive speedup >= 1.00 at all six concurrency levels.
  A FAIL is a valid empirical result and does not make the analysis fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from experiments import aggregate_results as agg


LANES = ("vanilla", "static", "adaptive")
WORKLOADS = ("W1", "W2", "W3")
CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 32)
SPEC_LANES = ("static", "adaptive")
W3_GATE_THRESHOLD = 1.0

M2_SUMMARY_COLUMNS = agg.SUMMARY_COLUMNS + [
    "experiment_id",
    "tpot_p50_min",
    "tpot_p50_max",
    "throughput_min",
    "throughput_max",
    "variance_ratio",
    "raw_sha256",
]


def _cell_id(lane: str, workload: str, concurrency: int) -> str:
    return f"{lane}__{workload}__c{concurrency}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_triplet_markers(root: Path) -> dict[str, Any]:
    """Validate the 18 triplet markers and return campaign identity."""
    errors: list[str] = []
    experiment_ids: set[str] = set()
    execution_shas: set[str] = set()
    expected_sha: dict[str, str] = {}

    for workload in WORKLOADS:
        for concurrency in CONCURRENCY_LEVELS:
            name = f"{workload}__c{concurrency}"
            path = root / "triplets" / f"{name}.json"
            if not path.exists():
                errors.append(f"{name}: missing triplet marker")
                continue

            marker = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                errors.append(f"{name}: marker must be a JSON object")
                continue
            if marker.get("status") != "COMPLETE_VALID":
                errors.append(f"{name}: status={marker.get('status')!r}")
                continue

            expected_cells = [_cell_id(lane, workload, concurrency) for lane in LANES]
            if (marker.get("workload"), marker.get("concurrency")) != (workload, concurrency):
                errors.append(
                    f"{name}: marker describes "
                    f"{marker.get('workload')!r}/c{marker.get('concurrency')!r}"
                )
                continue
            if sorted(marker.get("cells") or []) != sorted(expected_cells):
                errors.append(f"{name}: cells={marker.get('cells')!r}")
                continue

            for field, seen in (
                ("experiment_id", experiment_ids),
                ("execution_sha", execution_shas),
            ):
                value = marker.get(field)
                if not isinstance(value, str) or not value:
                    errors.append(f"{name}: missing {field}")
                seen.add(value)

            raw_sha = marker.get("raw_sha256")
            if not isinstance(raw_sha, dict):
                errors.append(f"{name}: raw_sha256 must be an object")
                continue

            for cell in expected_cells:
                if cell not in raw_sha:
                    errors.append(f"{name}: marker has no raw_sha256 for {cell}")
                    continue
                expected_sha[cell] = raw_sha[cell]

    if len(experiment_ids) > 1 or len(execution_shas) > 1:
        errors.append(
            "Triplets mix campaigns: "
            f"experiment_id={sorted(map(str, experiment_ids))}, "
            f"execution_sha={sorted(map(str, execution_shas))}"
        )

    if errors:
        raise ValueError("Invalid triplet markers: " + "; ".join(errors))

    return {
        "experiment_id": next(iter(experiment_ids)),
        "execution_sha": next(iter(execution_shas)),
        "raw_sha256": expected_sha,
    }


def _summarize_cell(raw_path: Path, lane: str) -> dict[str, Any]:
    records = agg._load_jsonl(raw_path)
    agg.validate_measured_records(records)
    measured = agg._measured_records(records)

    rows = agg._aggregate_measured_records(measured)
    if len(rows) != 1:
        raise ValueError(f"{raw_path}: expected exactly one cell, got {len(rows)}")
    row = rows[0]

    if lane in SPEC_LANES:
        alpha, tau = agg.aggregate_static_acceptance(measured)
        row["alpha"] = round(alpha, 9)
        row["tau"] = round(tau, 9)

    throughputs = agg.calculate_repeat_throughputs(measured)
    row["throughput_min"] = round(min(throughputs), 2)
    row["throughput_max"] = round(max(throughputs), 2)
    row["tpot_p50_min"] = row["_tpot_min"]
    row["tpot_p50_max"] = row["_tpot_max"]
    row["variance_ratio"] = round(
        (row["_tpot_max"] - row["_tpot_min"]) / row["tpot_p50"], 6
    )
    return row


def summarize_campaign(root: Path) -> list[dict[str, Any]]:
    """Validate provenance and aggregate all 54 cells with matched speedup."""
    root = Path(root)
    identity = load_triplet_markers(root)
    rows: list[dict[str, Any]] = []

    for workload in WORKLOADS:
        for concurrency in CONCURRENCY_LEVELS:
            for lane in LANES:
                cell = _cell_id(lane, workload, concurrency)
                raw_path = root / "cells" / cell / "raw.jsonl"
                if not raw_path.exists():
                    raise ValueError(f"{cell}: missing raw.jsonl at {raw_path}")

                actual = _sha256(raw_path)
                if actual != identity["raw_sha256"][cell]:
                    raise ValueError(
                        f"{cell}: raw.jsonl SHA256 {actual} does not match "
                        f"triplet marker {identity['raw_sha256'][cell]} "
                        "(CRLF checkout? extract with `git cat-file blob`)"
                    )

                row = _summarize_cell(raw_path, lane)
                if (row["lane"], row["workload"], row["concurrency"]) != (
                    lane, workload, concurrency
                ):
                    raise ValueError(
                        f"{cell}: raw identity is "
                        f"{row['lane']}/{row['workload']}/c{row['concurrency']}"
                    )
                if row["_experiment_id"] != identity["experiment_id"]:
                    raise ValueError(
                        f"{cell}: experiment_id {row['_experiment_id']!r} != "
                        f"campaign {identity['experiment_id']!r}"
                    )
                if row["commit"] != identity["execution_sha"]:
                    raise ValueError(
                        f"{cell}: commit {row['commit']!r} != "
                        f"execution_sha {identity['execution_sha']!r}"
                    )

                row["experiment_id"] = identity["experiment_id"]
                row["raw_sha256"] = actual
                rows.append(row)

    readiness = agg.check_m2_cells(rows)
    if readiness["status"] != "PASS":
        raise ValueError(f"M2 readiness check failed: {readiness}")

    return agg.apply_speedup_vs_vanilla(rows)


def evaluate_w3_gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Adaptive speedup >= 1.00 on W3 at every concurrency level."""
    by_key = {
        (row["lane"], row["workload"], row["concurrency"]): row for row in rows
    }
    cells = []

    for concurrency in CONCURRENCY_LEVELS:
        adaptive = by_key.get(("adaptive", "W3", concurrency))
        vanilla = by_key.get(("vanilla", "W3", concurrency))
        if adaptive is None or vanilla is None:
            raise ValueError(f"W3 gate requires adaptive and vanilla at c{concurrency}")

        speedup = float(adaptive["speedup_vs_vanilla"])
        cells.append({
            "concurrency": concurrency,
            "speedup": speedup,
            "vanilla_tpot_p50": vanilla["tpot_p50"],
            "adaptive_tpot_p50": adaptive["tpot_p50"],
            "adaptive_tpot_p50_min": adaptive.get("tpot_p50_min"),
            "adaptive_tpot_p50_max": adaptive.get("tpot_p50_max"),
            "pass": speedup >= W3_GATE_THRESHOLD,
        })

    return {
        "gate": "W3_ADAPTIVE_SPEEDUP_ALL_CONCURRENCY",
        "formula": "vanilla_tpot_p50 / adaptive_tpot_p50",
        "threshold": W3_GATE_THRESHOLD,
        "status": "PASS" if all(cell["pass"] for cell in cells) else "FAIL",
        "experiment_id": rows[0].get("experiment_id") if rows else None,
        "commit": rows[0].get("commit") if rows else None,
        "cells": cells,
    }


def write_m2_summary_csv(rows: Sequence[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=M2_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in M2_SUMMARY_COLUMNS})


def _print_table(rows: Sequence[dict[str, Any]], gate: dict[str, Any]) -> None:
    by_key = {(r["lane"], r["workload"], r["concurrency"]): r for r in rows}
    print(f"{'WL':<3} {'c':>3} | {'TPOT V':>7} {'TPOT S':>7} {'TPOT A':>7} "
          f"| {'S x':>6} {'A x':>6} | {'alpha S':>7} {'alpha A':>7}")
    for workload in WORKLOADS:
        for c in CONCURRENCY_LEVELS:
            v, s, a = (by_key[(lane, workload, c)] for lane in LANES)
            print(f"{workload:<3} {c:>3} | {v['tpot_p50']:7.2f} {s['tpot_p50']:7.2f} "
                  f"{a['tpot_p50']:7.2f} | {s['speedup_vs_vanilla']:6.3f} "
                  f"{a['speedup_vs_vanilla']:6.3f} | {s['alpha']:7.3f} {a['alpha']:7.3f}")
    print(f"\nW3 gate (Adaptive >= {gate['threshold']:.2f}x at all c): {gate['status']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2-04 three-lane summary and W3 gate")
    parser.add_argument("--campaign-root", required=True, type=Path,
                        help="Canonical campaign dir containing cells/ and triplets/.")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("experiments/results/m2_summary.csv"))
    parser.add_argument("--gate-json", type=Path,
                        default=Path("experiments/results/m2_w3_gate.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = summarize_campaign(args.campaign_root)
        gate = evaluate_w3_gate(rows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    write_m2_summary_csv(rows, args.output_csv)
    gate["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    _print_table(rows, gate)
    print(f"Wrote {len(rows)} rows -> {args.output_csv}; gate -> {args.gate_json}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
