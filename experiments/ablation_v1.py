"""[Adaptive v1] Ablation of the Lane 3 slowdown on GPU L4 (planning + analysis).

Lanes (experiments/modal_lanes.py): Vanilla, Static, Adaptive v0 and three variants
that add the fixes one by one (F1 draft floor, F2 warm-up guard, F3 uniform batch K).

    modal run -m experiments.modal_runner::ablation --server-run-id <deploy id> --campaign-id <id>

Evidence: experiments/runs/m2_02_v1/<campaign-id>/<lane>/<W>-c<c>/ and summary.json.
Speedup uses the M2-04 definition (vanilla_tpot_p50 / lane_tpot_p50). This is a new
campaign on L4; it never replaces the canonical M2-04 T4 results.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

LANES = ("vanilla", "static", "adaptive", "adaptive-f1", "adaptive-f12", "adaptive-v1", "adaptive-v2")
WORKLOADS = ("W1", "W2", "W3")
CONCURRENCY = (1, 4, 16)
REPEATS = 3
NUM_REQUESTS = 16
# Proposed acceptance criteria (to be confirmed by the PM): W3 at least 0.98x Vanilla,
# W1/W2 at most 3% slower than Static.
W3_MIN_SPEEDUP = 0.98
MIN_SPEED_VS_STATIC = 0.97
# L4 capacity can queue a cold server for many minutes (29/10: > 20 min); a lane that
# is still not ready after this is recorded as failed instead of aborting the campaign.
READY_TIMEOUT_S = 45 * 60
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def plan_cells(workloads: Sequence[str] = WORKLOADS, concurrency: Sequence[int] = CONCURRENCY,
               repeats: int = REPEATS, num_requests: int = NUM_REQUESTS) -> list[dict[str, Any]]:
    return [{"workload": w, "concurrency": c, "repeats": repeats, "num_requests": num_requests}
            for w in workloads for c in concurrency]


def run_cells(cells: Sequence[dict[str, Any]], run_one: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict]:
    """Run cells in order; a failing cell is recorded with its error and the rest still run."""
    results = []
    for cell in cells:
        try:
            result = dict(run_one(cell))
        except Exception as exc:  # one lane/cell failure must not lose the whole campaign
            result = {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}
        result["cell"] = cell
        results.append(result)
    return results


def campaign_dir(campaign_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(campaign_id or "") or ".." in campaign_id:
        raise ValueError(f"Invalid campaign id {campaign_id!r}")
    root = (REPO_ROOT / "experiments" / "runs" / "m2_02_v1").resolve()
    path = (root / campaign_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid campaign id {campaign_id!r}")
    return path


def add_speedups(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """speedup_vs_vanilla and speed_vs_static (baseline tpot / lane tpot) per cell."""
    rows = [dict(r) for r in rows]
    tpot = {(r["lane"], r["workload"], r["concurrency"]): r["tpot_p50"] for r in rows}
    for r in rows:
        key = (r["workload"], r["concurrency"])
        vanilla, static = tpot.get(("vanilla", *key)), tpot.get(("static", *key))
        r["speedup_vs_vanilla"] = vanilla / r["tpot_p50"] if vanilla else None
        r["speed_vs_static"] = static / r["tpot_p50"] if static else None
    return rows


def evaluate_criteria(rows: Sequence[dict[str, Any]], lane: str = "adaptive-v1") -> dict[str, Any]:
    checks = []
    for r in rows:
        if r["lane"] != lane:
            continue
        if r["workload"] == "W3":
            value, threshold, metric = r["speedup_vs_vanilla"], W3_MIN_SPEEDUP, "speedup_vs_vanilla"
        else:
            value, threshold, metric = r["speed_vs_static"], MIN_SPEED_VS_STATIC, "speed_vs_static"
        checks.append({"workload": r["workload"], "concurrency": r["concurrency"], "metric": metric,
                       "value": value, "threshold": threshold,
                       "pass": value is not None and value >= threshold})
    expected = {(w, c) for w in WORKLOADS for c in {r["concurrency"] for r in rows}}
    present = {(c["workload"], c["concurrency"]) for c in checks if c["value"] is not None}
    if not checks or not expected <= present:
        status = "INCOMPLETE"
    else:
        status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    return {"lane": lane, "status": status, "checks": sorted(checks, key=lambda c: (c["workload"], c["concurrency"]))}


def _epoch(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def cell_window(raw_records: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Measured time window of one cell (warm-up requests finish before it starts)."""
    starts = [_epoch(r["run_started_at_utc"]) for r in raw_records if r.get("run_started_at_utc")]
    ends = [_epoch(r["run_finished_at_utc"]) for r in raw_records if r.get("run_finished_at_utc")]
    return min(starts), max(ends)


def split_telemetry(lines: Iterable[dict[str, Any]],
                    windows: dict[tuple[str, int], tuple[float, float]]) -> dict[tuple[str, int], list[dict]]:
    parts: dict[tuple[str, int], list[dict]] = {key: [] for key in windows}
    for line in lines:
        for key, (start, end) in windows.items():
            if start <= line.get("finished_at", -1.0) <= end:
                parts[key].append(line)
                break
    return parts


def _merge(target: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        target[key] = target.get(key, 0) + value


def telemetry_summary(lines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    batch_k: dict[str, int] = {}
    budgets: dict[str, int] = {}
    for line in lines:
        _merge(reasons, line.get("reason_counts", {}))
        _merge(batch_k, line.get("batch_k_counts", {}))
        _merge(budgets, _count(line.get("tree_budgets", [])))
    total = sum(budgets.values())
    return {"requests": len(lines), "reason_counts": reasons, "batch_k_counts": batch_k,
            "k_share": {k: n / total for k, n in budgets.items()} if total else {}}


def _count(values: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def merge_jsonl(existing: str, new: str) -> str:
    """Lines of `existing`, then the lines of `new` not already present (a rerun of
    some cells must not drop the telemetry of the cells measured before)."""
    lines = [line for line in existing.splitlines() if line.strip()]
    seen = set(lines)
    lines += [line for line in new.splitlines() if line.strip() and line not in seen]
    return "".join(line + "\n" for line in lines)


def latest_attempt(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records of the last repeat series (r001, r002, ...) in a raw file. The harness appends
    to the volume file per bench run id, so a cell attempt that was cut off (L4 queueing,
    preemption) leaves its repeats in front of the complete rerun."""
    order: list[str] = []
    repeat: dict[str, str] = {}
    for r in records:
        if r["run_id"] not in repeat:
            order.append(r["run_id"])
            repeat[r["run_id"]] = r["repeat_id"]
    starts = [i for i, run_id in enumerate(order) if repeat[run_id] == "r001"]
    keep = set(order[starts[-1]:] if starts else order)
    return [r for r in records if r["run_id"] in keep]
