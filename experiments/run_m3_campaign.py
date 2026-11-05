"""Manual M3-05 controller. Never run `run` during Codex pre-flight."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from experiments import modal_lanes as lanes
from experiments.validate_m3 import validate_cell

ROOT = Path(__file__).resolve().parents[1]
GRID = [(lane, workload, c) for lane in lanes.LANES
        for workload in lanes.BENCH_WORKLOADS for c in (1, 2, 4, 8, 16, 32)]


def plan(campaign_id: str, sha: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise ValueError("Invalid campaign_id")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A full committed execution SHA is required")
    return {"campaign_id": campaign_id, "commit": sha, "kind": "confirmatory",
            "warmup": 3, "lifecycle": "fresh-server-per-cell", "repeats": 5, "num_requests": 32,
            "cells": [{"key": f"{lane}__{workload}__c{c}", "lane": lane,
                       "workload": workload, "concurrency": c} for lane, workload, c in GRID]}


def save_plan(record: dict) -> Path:
    out = ROOT / "experiments/runs/m3_05" / record["campaign_id"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    content = (json.dumps(record, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError("Campaign plan differs from frozen existing plan")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(content)
    return out


def attempts(record: dict, cell: dict) -> list[Path]:
    prefix = f"{record['campaign_id']}-{cell['lane']}-{cell['workload']}-c{cell['concurrency']}-a"
    return sorted((ROOT / "experiments/runs/m3_01").glob(prefix + "[0-9][0-9][0-9]"))


def selected_attempt(record: dict, cell: dict) -> Path | None:
    valid = []
    for out in attempts(record, cell):
        if (out / "validation_failure.json").exists():
            continue
        if not (out / "bench.json").exists():
            continue
        verified = validate_cell(out)
        if verified["commit"] != record["commit"]:
            raise ValueError("Campaign/attempt source mismatch")
        receipt = json.loads((out / "submission.json").read_text(encoding="utf-8"))
        if (receipt["lane"], receipt["cell"]["workload"], receipt["cell"]["concurrency"]) != (
                cell["lane"], cell["workload"], cell["concurrency"]):
            raise ValueError("Attempt in wrong scientific cell")
        valid.append(out)
    if len(valid) > 1:
        raise ValueError(f"Duplicate complete canonical cell: {cell['key']}")
    return valid[0] if valid else None


def command(*args: str, env: dict | None = None) -> None:
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, check=True)


def stop() -> None:
    apps = json.loads(subprocess.check_output(
        [sys.executable, "-m", "modal", "app", "list", "--json"], cwd=ROOT, text=True))
    for app in apps:
        if app["description"] == lanes.APP_NAME and app["state"] != "stopped":
            command("modal", "app", "stop", app["app_id"], "-y")


def drive(record: dict, retry_failed: bool = False) -> None:
    import os
    from experiments import modal_runner as runner

    if runner._committed_bench_source() != record["commit"]:
        raise ValueError("Checkout must match the reviewed campaign SHA")
    save_plan(record)
    for cell in record["cells"]:
        if selected_attempt(record, cell):
            print(f"COLLECTED {cell['key']}; skipping", flush=True)
            continue
        history = attempts(record, cell)
        pending = history[-1] if history else None
        state = runner.bench_status(pending.name) if pending else "NOT_SUBMITTED"
        if pending and state == "NOT_SUBMITTED" and (pending / "spawn_attempt.json").exists():
            raise RuntimeError("Spawn outcome unknown; recover the existing call; NEVER auto-resubmit")
        if state == "FAILED" or (state == "NOT_SUBMITTED" and pending):
            # Never infer terminal remote failure from a network error.
            if not retry_failed:
                raise RuntimeError(f"{pending}: {state}; inspect evidence, then use --retry-failed")
            stop()
            pending = None
        if pending is None:
            number = len(history) + 1
            if number > 999:
                raise ValueError("Too many attempts")
            run_id = (f"{record['campaign_id']}-{cell['lane']}-{cell['workload']}"
                      f"-c{cell['concurrency']}-a{number:03}")
            stop()  # confirmed stop, before deploying a new cell's server
            env = {**os.environ, "PTIT_RUN_ID": run_id, "PTIT_LANES": cell["lane"],
                   "PTIT_KEEP_WARM": "0", "PTIT_SCALEDOWN_SECONDS": "60"}
            command("modal", "deploy", "-m", "experiments.modal_runner", env=env)
            runner.bench_submit(lane=cell["lane"], workload=cell["workload"],
                                concurrency=cell["concurrency"], repeats=5, num_requests=32,
                                server_run_id=run_id, bench_run_id=run_id)
            pending = ROOT / "experiments/runs/m3_01" / run_id
        while True:
            state = runner.bench_status(pending.name)
            print(f"{state} {cell['key']} {pending.name}", flush=True)
            if state in ("COMPLETED", "COLLECTED"):
                receipt = json.loads((pending / "submission.json").read_text(encoding="utf-8"))
                runner.bench_collect(receipt["call_id"])
                try:
                    result = validate_cell(pending)
                except (ValueError, KeyError, OSError) as exc:
                    failure = pending / "validation_failure.json"
                    if not failure.exists():
                        failure.write_bytes((json.dumps({"status": "INVALID-PROTOCOL", "error": str(exc)}) + "\n").encode())
                    stop()
                    raise
                path = pending / "validation.json"
                if not path.exists():
                    path.write_bytes((json.dumps(result, indent=2) + "\n").encode())
                stop()
                break
            if state == "FAILED":
                stop()
                raise RuntimeError(f"Failed attempt preserved: {pending}; inspect before --retry-failed")
            if state == "NOT_SUBMITTED":
                raise RuntimeError(f"No call ID: {pending}; inspect before explicit new attempt")
            # Laptop interruption exits this loop without cancelling a live remote call.
            time.sleep(15)


def validate_campaign(record: dict) -> list[Path]:
    selected = [selected_attempt(record, cell) for cell in record["cells"]]
    if len(selected) != 72 or any(path is None for path in selected):
        missing = [c["key"] for c, p in zip(record["cells"], selected) if p is None]
        raise ValueError(f"Incomplete campaign: {missing}")
    instance_ids = [validate_cell(path)["server_instance_id"] for path in selected]
    if len(set(instance_ids)) != 72:
        raise ValueError("Server instance reused between scientific cells")
    return selected


def aggregate(record: dict, destination: Path) -> None:
    from experiments import aggregate_results as agg

    paths = validate_campaign(record)
    rows = []
    for path in paths:
        raw_path = path / "raw.vram.jsonl"
        if not raw_path.exists():
            raise ValueError(f"VRAM evidence required for final aggregation: {path}")
        raw = agg._load_jsonl(raw_path)
        row = agg.aggregate_raw_jsonl(raw_path)[0]
        if row["lane"] != "vanilla":
            row["alpha"], row["tau"] = agg.aggregate_static_acceptance(raw)
        row["attempt_id"] = path.name
        # Pair by the frozen campaign, validated same-SHA grid and workload hash.
        # Original per-attempt experiment IDs remain intact in raw and the index.
        row["_experiment_id"] = record["campaign_id"]
        rows.append(row)
    rows = agg.apply_speedup_vs_vanilla(rows)
    result = {"campaign": record, "scientific_status": "VALID", "engineering_gate": "NOT_DEFINED",
              "selected_attempts": [str(p.relative_to(ROOT)) for p in paths],
              "all_attempts": {c["key"]: [p.name for p in attempts(record, c)] for c in record["cells"]},
              "cells": rows}
    content = (json.dumps(result, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != content:
        raise FileExistsError(f"Conflicting summary: {destination}")
    if not destination.exists():
        destination.write_bytes(content)


def figures(summary: Path, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = json.loads(summary.read_text(encoding="utf-8"))
    rows = record["cells"]
    if len(rows) != 72 or record["scientific_status"] != "VALID":
        raise ValueError("Figures require a validated full campaign summary")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, metrics in (("F1_m3_tpot_throughput_4lanes.png", ("tpot_p50", "throughput")),
                              ("F2_m3_speedup_vs_vanilla.png", ("speedup_vs_vanilla",))):
        target = destination / filename
        if target.exists():
            raise FileExistsError(target)
        fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 4 * len(metrics)), squeeze=False)
        for i, metric in enumerate(metrics):
            for j, workload in enumerate(lanes.BENCH_WORKLOADS):
                ax = axes[i, j]
                for lane in lanes.LANES:
                    values = sorted((r for r in rows if r["lane"] == lane and r["workload"] == workload),
                                    key=lambda r: r["concurrency"])
                    ax.plot([r["concurrency"] for r in values], [r[metric] for r in values], marker="o", label=lane)
                ax.set_xscale("log", base=2)
                ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
                ax.set_xlabel("Concurrency")
                ax.set_title(workload)
                ax.set_ylabel({"tpot_p50": "TPOT p50 (ms)", "throughput": "Output tokens/s",
                               "speedup_vs_vanilla": "Speedup vs Vanilla (TPOT ratio)"}[metric])
                if metric == "speedup_vs_vanilla":
                    ax.axhline(1, linestyle="--", color="black")
                ax.grid(alpha=0.3)
                ax.legend()
        fig.suptitle(f"M3-05: {record['campaign']['campaign_id']} — repeat medians; negative results retained")
        fig.tight_layout()
        fig.savefig(target, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "validate", "aggregate", "figures", "stop"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    record = plan(args.campaign_id, args.execution_sha)
    if args.action == "plan":
        print(json.dumps(record, indent=2))  # read-only; does not create an evidence directory
        return
    out = save_plan(record)
    summary = ROOT / "experiments/results/m3_sweep_4lanes_summary.json"
    if args.action == "run":
        drive(record, args.retry_failed)
    elif args.action == "status":
        from experiments import modal_runner as runner
        for cell in record["cells"]:
            history = attempts(record, cell)
            state = "COLLECTED" if selected_attempt(record, cell) else (
                runner.bench_status(history[-1].name) if history else "NOT_SUBMITTED")
            print(cell["key"], state)
    elif args.action == "validate":
        print(f"VALID: {len(validate_campaign(record))} cells, 11520 measured requests")
    elif args.action == "aggregate":
        aggregate(record, summary)
    elif args.action == "figures":
        figures(summary, ROOT / "experiments/results/figures/m3")
    else:
        stop()


if __name__ == "__main__":
    main()
