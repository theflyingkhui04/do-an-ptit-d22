"""Thin M2-04 interleaving wrapper around the existing switcher and harness.

Default: print the CPU-only command plan. GPU execution requires --execute.
No resume or analysis pipeline is introduced; each attempt needs a fresh path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "W1": "experiments/configs/w1/node1_chat_requests.jsonl",
    "W2": "experiments/configs/w2_v2/chat_requests.jsonl",
    "W3": "experiments/configs/w3_v2/chat_requests.jsonl",
}
LANES = (
    ("vanilla", "lane1", 8001, "qwen2.5-7b-vanilla"),
    ("static", "lane2", 8002, "qwen2.5-7b-static-draft"),
    ("adaptive", "lane3", 8003, "qwen2.5-7b-adaptive"),
)


def build_plan(output: Path, repeats: int = 5) -> list[dict]:
    if isinstance(repeats, bool) or repeats < 5:
        raise ValueError("M2 requires >=5 measured repeats per cell")
    output = output.resolve()
    plan = []
    for workload, artifact in WORKLOADS.items():
        for concurrency in (1, 2, 4, 8, 16, 32):
            for lane, switch, port, model in LANES:
                cell = f"{lane}__{workload}__c{concurrency}"
                cell_dir = output / cell
                plan.append({
                    "cell": cell,
                    "switch": ["bash", str(ROOT / "scripts/switch_lane.sh"), switch],
                    "harness": [
                        sys.executable, str(ROOT / "experiments/benchmark_harness.py"),
                        "--lane", lane, "--workload", workload,
                        "--endpoint", f"http://127.0.0.1:{port}/v1/chat/completions",
                        "--model", model, "--workload-file", str(ROOT / artifact),
                        "--sample-policy", "cycle-rotate",
                        "--concurrency", str(concurrency), "--repeats", str(repeats),
                        "--num-requests", "32", "--warmup-requests", "3",
                        "--experiment-id", output.name,
                        "--raw-out", str(cell_dir / "raw.jsonl"),
                        "--summary-csv", str(output / "harness_header.csv"),
                    ],
                })
    return plan


def verify_execution(execution_sha: str) -> None:
    """Fail before lane switching on unfrozen code or noncanonical bytes."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if len(execution_sha) != 40 or git("rev-parse", "HEAD") != execution_sha:
        raise ValueError("--execution-sha must match the full reviewed HEAD SHA")
    if git("status", "--porcelain"):
        raise ValueError("Canonical execution requires a clean reviewed checkout")

    from experiments.workload_adapter import (
        load_workload_rows, validate_manifest_consistency,
        validate_workload_label, verify_workload_integrity,
    )
    for workload, artifact in WORKLOADS.items():
        path = ROOT / artifact
        integrity = verify_workload_integrity(path)
        rows = load_workload_rows(path)
        validate_manifest_consistency(integrity, rows)
        validate_workload_label(workload, rows[0]["workload_id"])


def execute(plan: list[dict], output: Path, execution_sha: str) -> None:
    if output.exists():
        raise FileExistsError(f"Evidence path already exists: {output}")
    verify_execution(execution_sha)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump({"execution_sha": execution_sha, "cells": plan}, stream, indent=2)
    for item in plan:
        cell_dir = output / item["cell"]
        cell_dir.mkdir()
        env = dict(os.environ, LANE_LOG_DIR=str(cell_dir / "server"),
                   ADAPTIVE_TELEMETRY_PATH=str(cell_dir / "adaptive_telemetry.jsonl"))
        # The switcher waits for health and releases the previous lane's GPU.
        # Its log override prevents its historical default logs being replaced.
        for name in ("switch", "harness"):
            with (cell_dir / f"{name}.log").open("x", encoding="utf-8") as log:
                subprocess.run(item[name], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    # Readiness evidence only: no final analysis CSV or W3 gate assertion.
    with (output / "readiness.json").open("x", encoding="utf-8") as report:
        subprocess.run(
            [sys.executable, "-m", "experiments.aggregate_results", "--check-m2",
             "--raw", *(str(output / item["cell"] / "raw.jsonl") for item in plan)],
            cwd=ROOT, stdout=report, check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh evidence directory; never reused or overwritten")
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--execute", action="store_true",
                        help="Switch live servers and run the GPU campaign")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        plan = build_plan(output, args.repeats)
        if args.execute:
            execute(plan, output, args.execution_sha)
        else:
            print(json.dumps({"execution_sha": args.execution_sha, "cells": plan}, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
