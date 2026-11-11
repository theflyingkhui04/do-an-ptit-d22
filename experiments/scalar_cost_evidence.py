"""CPU-only validation of proposed scalar retirement-window evidence.

No collector, GPU launch, profile publication or controller policy. See
docs/runbooks/m4-02-04-scalar-cost-contract.md. Passing validates data structure
and arithmetic, never the truth of the runtime observations supplied to it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from experiments import modal_lanes as lanes
from experiments.scalar_context import VERSION as CONTEXT_VERSION

PROTOCOL = "scalar-retirement-window-v1"
BOUNDARY = "enginecore-post-update-to-post-update"
IDENTITY_FIELDS = (
    "gpu_name", "sm_architecture", "driver", "cuda", "torch", "vllm",
    "target_model", "target_revision", "draft_model", "draft_revision",
    "dtype", "quantization", "effective_quantization", "draft_method",
    "scheduler_cls", "scheduler_base_cls", "runtime_config_sha256", "protocol",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_identity(observed, expected):
    for field in IDENTITY_FIELDS:
        require(isinstance(expected.get(field), str) and bool(expected[field]),
                f"Missing expected identity: {field}")
        require(observed.get(field) == expected[field], f"Identity mismatch: {field}")
    for field, value in (
        ("protocol", PROTOCOL), ("vllm", lanes.VLLM_VERSION),
        ("target_model", lanes.TARGET_MODEL), ("target_revision", lanes.TARGET_REVISION),
        ("draft_model", lanes.DRAFT_MODEL), ("draft_revision", lanes.DRAFT_REVISION),
        ("draft_method", "draft_model"),
        ("scheduler_base_cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"),
    ):
        require(observed[field] == value, f"Noncanonical identity: {field}")
    require(re.fullmatch(r"[0-9a-f]{64}", observed["runtime_config_sha256"]) is not None,
            "Invalid runtime configuration digest")
    hardware = {
        "NVIDIA L4": ("sm_89", "bfloat16", "awq_marlin"),
        "Tesla T4": ("sm_75", "half", "awq"),
    }
    require(observed["gpu_name"] in hardware, "Unsupported profiling hardware")
    require(tuple(observed[key] for key in ("sm_architecture", "dtype", "quantization"))
            == hardware[observed["gpu_name"]], "Hardware/runtime profile mismatch")


def validate_window(record, cell, cycles):
    """Validate a steady decode cohort; do not impute missing worker metadata."""
    require(record.get("timing_boundary") == BOUNDARY, "Wrong timing boundary")
    require(record.get("clock") == "perf_counter_ns", "Wrong clock")
    context_source = record.get("context_source")
    require(context_source in (None, "measured", "reconstructed"), "Forecast context is not execution evidence")
    if context_source == "reconstructed":
        require(record.get("context_reconstruction_version") == CONTEXT_VERSION,
                "Unknown context reconstruction version")
    require(bool(record.get("clock_domain")), "Missing process/boot clock domain")
    require(integer(record.get("start_ns")), "Invalid start timestamp")
    events = record.get("retirements", [])
    require(len(events) == cycles, "Missing/extra retirement")
    budgets = cell["request_budgets"]
    k = cell["shared_k"]
    require(type(k) is int and k in (0, 2, 3, 4), "Invalid K label")
    require(bool(budgets) and all(type(v) is int and v in (0, 2, 3, 4) and v <= k
                                for v in budgets), "Invalid request budgets")
    require(max(budgets) == k, "Shared K must match maximum request budget")
    low, high = cell["context_bin"]
    require(integer(low) and integer(high, 1) and low < high, "Invalid context bin")
    previous = record["start_ns"]
    request_budgets = None
    event_ids = set()
    emitted = 0
    for event in events:
        require(integer(event.get("end_ns")) and event["end_ns"] > previous,
                "Nonpositive/unordered retirement duration")
        previous = event["end_ns"]
        event_id = event.get("verifier_id")
        require(isinstance(event_id, str) and event_id and event_id not in event_ids,
                "Missing/duplicate verifier ID")
        event_ids.add(event_id)
        require(type(event.get("shared_k")) is int and event["shared_k"] == k,
                "K changed inside window")
        require(event.get("producer_rows") == len(budgets), "Producer batch mismatch")
        require(event.get("proposer_calls") == 1, "Missing/extra proposer call")
        if k == 0:
            require(integer(event.get("draft_forward_calls"), 1),
                    "K=0 draft synchronization unobserved")
        rows = event.get("requests", [])
        require(len(rows) == len(budgets), "Worker batch mismatch")
        current_ids = [row.get("request_id") for row in rows]
        require(all(isinstance(v, str) and v for v in current_ids)
                and len(set(current_ids)) == len(rows), "Invalid request identities")
        if request_budgets is None:
            request_budgets = dict(zip(current_ids, budgets))
        require(set(current_ids) == set(request_budgets), "Cohort changed inside window")
        for row in rows:
            budget = request_budgets[row["request_id"]]
            require(row.get("prefill") is False and row.get("terminal") is False
                    and row.get("stale") is False, "Non-steady decode row")
            require(type(row.get("effective_k")) is int and row["effective_k"] == budget,
                    "Request budget mismatch")
            require(integer(row.get("context_tokens"))
                    and low <= row["context_tokens"] < high, "Context outside cell")
            if context_source == "reconstructed":
                require(row.get("pre_execution_context") == row["context_tokens"]
                        and row.get("verification_query_length") == row.get("query_tokens")
                        and integer(row.get("raw_valid_sampled_count"))
                        and row.get("corrected_post_execution_context") ==
                        row["context_tokens"] + row["raw_valid_sampled_count"],
                        "Inconsistent reconstructed pre/query/post contexts")
            drafted, accepted = row.get("drafted"), row.get("accepted")
            require(integer(drafted) and drafted == budget, "Unsaturated/invalid drafts")
            require(integer(accepted) and accepted <= drafted, "Invalid acceptance count")
            require(integer(row.get("query_tokens"), 1)
                    and row["query_tokens"] == drafted + 1, "Query shape mismatch")
            require(integer(row.get("emitted")) and row["emitted"] == accepted + 1,
                    "Emitted count disagrees with nonterminal scalar output")
            emitted += row["emitted"]
    require(record.get("end_ns") == previous, "End boundary mismatch")
    require(record.get("tree_execution_verified") is False, "Tree execution overclaim")
    return (previous - record["start_ns"]) / 1_000_000, emitted


def validate_run(manifest, records):
    """Report incomplete cells without dropping failed raw attempts.

    Even a complete report is not a consumer profile: paired-reference shape
    equivalence and collector non-perturbation still require technical review.
    """
    validate_identity(manifest["observed_identity"], manifest["expected_identity"])
    require(re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_sha", "")) is not None,
            "Invalid Git SHA")
    require(bool(manifest.get("timestamp_utc")) and bool(manifest.get("run_id")),
            "Missing run provenance")
    warmup, repeats, cycles = (manifest[name] for name in ("warmup", "repeats", "cycles"))
    require(integer(warmup, 5) and integer(repeats, 50) and integer(cycles, 8),
            "Profiling requires >=5 warmup cohorts, >=50 measured cohorts, >=8 intervals")
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    require(bool(cells) and len(cells) == len(manifest["cells"]), "Duplicate/empty cells")
    for cell in cells.values():
        reference = cells.get(cell.get("reference_cell_id"))
        require(reference is not None and reference["shared_k"] == 0
                and reference["request_budgets"] == [0] * len(cell["request_budgets"])
                and reference["context_bin"] == cell["context_bin"],
                "Missing matched K=0 reference cell")
    seen, samples, failures = set(), {key: [] for key in cells}, []
    cohort_ids, clock_ends = set(), {}
    for record in records:
        require(record.get("run_id") == manifest["run_id"]
                and record.get("git_sha") == manifest["git_sha"], "Mixed run provenance")
        key, phase, index = record.get("cell_id"), record.get("phase"), record.get("repeat")
        require(key in cells and phase in ("warmup", "measured") and integer(index),
                "Invalid cell/phase/repeat")
        require(index < (warmup if phase == "warmup" else repeats), "Repeat outside plan")
        require((key, phase, index) not in seen, "Duplicate attempt; retries need new run IDs")
        seen.add((key, phase, index))
        cohort = record.get("cohort_id")
        require(isinstance(cohort, str) and cohort and cohort not in cohort_ids,
                "Missing/reused independent cohort ID")
        cohort_ids.add(cohort)
        if record.get("status") != "ok":
            require(record.get("status") in ("error", "oom", "timeout", "unsupported_shape")
                    and bool(record.get("error")), "Unexplained failed record")
            failures.append({"cell_id": key, "phase": phase, "repeat": index,
                             "status": record["status"], "error": record["error"]})
            continue
        duration, emitted = validate_window(record, cells[key], cycles)
        domain = record["clock_domain"]
        require(record["start_ns"] >= clock_ends.get(domain, 0),
                "Overlapping/reordered windows in one clock domain")
        clock_ends[domain] = record["end_ns"]
        if phase == "measured":
            samples[key].append((duration, emitted))
    expected = {(key, phase, index) for key in cells
                for phase, count in (("warmup", warmup), ("measured", repeats))
                for index in range(count)}
    summary = []
    for key, values in samples.items():
        row = {"cell_id": key, "successful_measured_cohorts": len(values)}
        if values:
            costs = [duration / cycles for duration, _ in values]
            median = statistics.median(costs)
            row.update(median_ms_per_retired_batch=median,
                       p90_ms_per_retired_batch=sorted(costs)[math.ceil(.9 * len(costs)) - 1],
                       min_ms=min(costs), max_ms=max(costs),
                       variance_ratio=(max(costs) - min(costs)) / median,
                       mean_emitted_tokens_per_batch=sum(n for _, n in values) / (cycles * len(values)),
                       pooled_emitted_tokens_per_ms=sum(n for _, n in values) / sum(t for t, _ in values))
        summary.append(row)
    return {"protocol": PROTOCOL, "artifact_kind": "scalar_cost_readiness_report",
            "consumer_profile": False, "complete": seen == expected and not failures,
            "missing_attempts": sorted(expected - seen), "failures": failures,
            "summary": summary}


def json_line(value):
    """No nonfinite numbers; caller must append/fsync raw records before validation."""
    return json.dumps(value, allow_nan=False, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (args.run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    report = validate_run(manifest, records)
    print(json_line(report), end="")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
