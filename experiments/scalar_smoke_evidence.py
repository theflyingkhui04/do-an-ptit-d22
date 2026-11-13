"""Offline, non-consumer validation for exactly two tiny instrumentation cells.

Reuses the existing identity/window validator without relaxing its full-profile
sample-size requirements. No cost aggregation or profitability calculation.
"""
import hashlib
import json
import re
from pathlib import Path

from experiments.scalar_context import (
    ContextRow, ExecutionContextSnapshot, MaterializedCounts, reconstruct, reference_match,
)
from experiments.scalar_cost_evidence import validate_identity, validate_window, require

PINNED_SOURCES = (
    "v1/engine/core.py", "v1/executor/uniproc_executor.py", "v1/outputs.py",
    "v1/core/sched/scheduler.py", "v1/core/sched/async_scheduler.py", "v1/core/sched/output.py",
    "v1/core/sched/utils.py", "v1/worker/gpu_model_runner.py", "v1/worker/gpu_input_batch.py",
    "v1/spec_decode/utils.py", "v1/spec_decode/llm_base_proposer.py",
    "v1/spec_decode/draft_model.py", "v1/sample/rejection_sampler.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(events):
    previous, result = None, []
    for event in events:
        if "preparation_context_snapshot" not in event:
            continue
        saved = event["preparation_context_snapshot"]
        snapshot = ExecutionContextSnapshot(**dict(saved,
            rows=tuple(ContextRow(**row) for row in saved["rows"]),
            lifecycle_flags=tuple(saved["lifecycle_flags"])))
        counts = MaterializedCounts(snapshot.execution_id, tuple(
            (rid, event["incarnations"][rid], value)
            for rid, value in event["raw_sampled_counts_before_update"]))
        terminal = tuple(row["request_id"] for row in event["context_reconstruction"]["rows"]
                         if not row["next_execution_eligible"])
        value = reconstruct(snapshot, counts, previous, terminal_requests=terminal)
        require(list(value.pre_contexts) == event["worker"]["contexts"], "Context recurrence mismatch")
        previous = value
        result.append(value)
    return result


def validate(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["artifact_kind"] == "scalar_observer_tiny_smoke", "Wrong artifact kind")
    require(manifest["gpu_execution_requested_from_codex"] is False, "Wrong execution route")
    require(manifest["cells"] == [0, 2] and manifest["worker_batch"] == 1
            and manifest["intervals"] == 3, "Expanded smoke plan")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["git_sha"]) is not None, "Wrong Git SHA")
    require(manifest["status"] == "COLLECTED" and not (root / "failure.json").exists(), "Run failure retained")
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    required = {"manifest.json", "pre_gpu_tests.json", "pre_gpu_tests.log", "config.json",
                "source_manifest.json", "environment.json", "raw.jsonl"}
    for k in (0, 2):
        required.update(f"k{k}/{name}" for name in (
            "cell.json", "observer.json", "fixture.json", "environment.json", "stdout-stderr.log"))
    require(required.issubset(hashes), "Missing mandatory artifact hashes")
    for name, digest in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), "Unsafe/missing artifact path")
        require(sha256(path) == digest, "Artifact hash mismatch: " + name)
    gate = json.loads((root / "pre_gpu_tests.json").read_text())
    require(gate["exit_code"] == 0 and gate["passed"] >= 230 and gate["skipped"] == 0,
            "Pre-GPU test gate incomplete")
    records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    require(len(records) == 2, "Missing/extra smoke cell")
    trajectories = []
    identities = []
    fixtures = []
    for k, record in zip((0, 2), records):
        folder = root / f"k{k}"
        cell = json.loads((folder / "cell.json").read_text())
        environment = json.loads((folder / "environment.json").read_text())
        validate_identity(environment["identity"], environment["identity"])
        require(environment["identity"]["gpu_name"] == "NVIDIA L4", "Wrong smoke hardware")
        require(environment["identity"]["torch"] == "2.13.0+cu130"
                and environment["identity"]["cuda"] == "13.0", "Pinned CUDA/PyTorch mismatch")
        identities.append(environment["identity"])
        require(cell["status"] == "PASS" and cell["k"] == k, "Failed cell")
        require(cell["enabled_output_token_ids"] == cell["disabled_output_token_ids"],
                "Observer enabled/disabled output mismatch")
        data = json.loads((folder / "observer.json").read_text())
        require(not data["failures"] and not data["overflow"], "Observer failures retained")
        source = json.loads((root / "source_manifest.json").read_text())
        require(source["git_sha"] == manifest["git_sha"], "Source snapshot identity mismatch")
        require(set(environment["installed_source_hashes"]) == {"vllm/" + path for path in PINNED_SOURCES},
                "Missing installed source hashes")
        for name, digest in environment["installed_source_hashes"].items():
            require(source["pinned_vllm_sources"][name]["sha256"] == digest, "Installed source evidence mismatch")
        values = replay(data["events"])
        by_id = {value.snapshot.execution_id: value for value in values}
        fixtures.append(json.loads((folder / "fixture.json").read_text()))
        fixture = fixtures[-1]
        require(hashlib.sha256(fixture["prompt"].encode()).hexdigest() == fixture["prompt_sha256"]
                and hashlib.sha256(json.dumps(fixture["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
                    == fixture["tokenized_prompt_sha256"], "Fixture hash mismatch")
        require(record["cell_id"] == f"k{k}" and record["status"] == "ok", "Wrong/failed raw cell")
        validate_window(record, dict(shared_k=k, request_budgets=[k], context_bin=[0, 4096]), 3)
        trajectory = []
        facts = {fact["verifier_id"]: fact for fact in data["execution_facts"]}
        events = {event.get("verifier_id"): event for event in data["events"]}
        anchor = events[record["anchor_verifier_id"]]
        require(anchor["status"] == "eligible" and anchor["end_ns"] == record["start_ns"],
                "Missing/ineligible retirement anchor")
        selected_ids = [record["anchor_verifier_id"], *[r["verifier_id"] for r in record["retirements"]]]
        positions = [next(i for i, event in enumerate(data["events"])
                          if event.get("verifier_id") == verifier) for verifier in selected_ids]
        require(positions == list(range(positions[0], positions[0] + 4)), "Window spans lifecycle break")
        eligible = [e["verifier_id"] for e in data["events"] if e["status"] == "eligible" and e["scheduled_k"] == k]
        require(selected_ids == eligible[4:8], "Wrong tiny settling/window selection")
        for retirement in record["retirements"]:
            verifier = retirement["verifier_id"]
            event, fact = events[verifier], facts[verifier]
            require(event["status"] == "eligible" and fact["label"] == "DECODE_ELIGIBLE",
                    "Ineligible retirement in window")
            require(fact["worker_rows"] == fact["proposer_rows"] == 1, "Actual worker B != 1")
            require(fact["target_forward_calls"] == 1, "Actual target dispatch unobserved")
            require(fact["materialized_output_id"] == event["model_output_object_id"]
                    and fact["schedule_object_id"] == event["schedule_object_id"]
                    and fact["execution_future_id"] is not None
                    and fact["output_future_id"] is not None, "Live future/S/O pairing mismatch")
            require(event["planned_k"] == event["scheduled_k"] == fact["proposer_k"]
                    == fact["assigned_proposer_k"] == k, "Requested/proposer K mismatch")
            rid = fact["request_ids"][0]
            require(event["planned_per_request"] == {rid: k}
                    and event["placeholder_counts"] == {rid: k}
                    and fact["draft_output_shape"] == [1, k], "K_i/placeholder/draft width mismatch")
            require(fact["invalid_slots"] == [0] and fact["queries"] == [k + 1],
                    "Unsaturated/invalid verification")
            require(event["emitted"] == sum(event["emitted_by_request"].values())
                    == sum(row["emitted"] for row in retirement["requests"]), "Useful yield mismatch")
            require(retirement["end_ns"] == event["end_ns"]
                    and retirement["proposer_calls"] == fact["proposer_calls"]
                    and retirement["draft_forward_calls"] == fact["draft_forward_calls"], "Unbound retirement facts")
            actual = by_id[verifier]
            require(retirement["requests"][0]["context_tokens"] == actual.pre_contexts[0]
                    and retirement["requests"][0]["query_tokens"] == actual.snapshot.rows[0].query_length,
                    "Raw window context/shape differs from reconstruction")
            trajectory.append(by_id[verifier])
        trajectories.append(trajectory)
    require(fixtures[0] == fixtures[1], "Cross-cell fixture/tokenization mismatch")
    require(identities[0] == identities[1], "Cross-cell runtime identity mismatch")
    matching = reference_match(*trajectories, identity_equal=True)
    return {"offline_validation": "PASS", "consumer_profile": False,
            "verdict": "LIVE OBSERVER VERIFIED", "context_matching": matching,
            "context_matching_reason": ("Exact mapped corrected context trajectories match"
                if matching == "MATCHED_REFERENCE" else "Exact mapped corrected context trajectories or padding differ"),
            "profitability": "NOT EVALUATED — INSTRUMENTATION SMOKE ONLY",
            "full_grid": "NOT RUN — AWAITING SMOKE REVIEW",
            "perturbation": "No device read/wait added; CPU metadata overhead retained. "
                            "Enabled/disabled output equality is a limited structural check, not zero perturbation."}
