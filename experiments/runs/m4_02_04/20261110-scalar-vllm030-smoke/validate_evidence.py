"""Validate saved real-engine evidence; does not simulate the engine."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from experiments.bench_source import snapshot

run = ROOT / "attempt2"
rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
aggregates = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
env = json.loads((run / "environment.json").read_text())
assert env["vllm"] == "0.30.0" and env["cuda_available"]
worker_hashes = json.loads((run / "worker_source_hashes.json").read_text())
temp, sha = snapshot(REPO, "a6cd6ad5cee5374738f890c270ada4ca01f0946b")
with temp:
    for name, digest in worker_hashes.items():
        relative = name.removeprefix("/root/project/") if name.startswith("/root/project/") else name.removeprefix("/root/")
        assert hashlib.sha256((Path(temp.name) / relative).read_bytes()).hexdigest() == digest, name
assert len(aggregates) == 3
assert all(r["tree_execution_verified"] is False for r in rows + aggregates)
plans = {(r["request_id"], r["plan_sequence"]): r for r in rows if r["record_type"] == "planned_decision"}
scheduled = [r for r in rows if r["record_type"] == "scheduled_budget"]
feedback = [r for r in rows if r["record_type"] == "verification_feedback"]
joined = 0
for r in scheduled:
    assert r["scheduled_runtime_mode"] == "linear_dynamic_k"
    if r["plan_sequence"] is not None:
        p = plans[r["request_id"], r["plan_sequence"]]
        assert p["batch_scalar_budget"] == r["scheduled_scalar_k"]
        assert r["next_request_placeholder_budget"] <= p["scalar_runtime_budget"]
        joined += 1
assert all(r["source_plan_sequence"] is None and r["source_schedule_sequence"] is None for r in feedback)
summaries = []
for aggregate in aggregates:
    rid = aggregate["request_id"]
    rr = [r for r in rows if r["request_id"] == rid]
    ff = [r for r in feedback if r["request_id"] == rid]
    assert ff and all(r["recorded_at"] <= aggregate["finished_at"] for r in rr)
    summaries.append({"request_id": rid, "scheduled_k": sorted({r["scheduled_scalar_k"] for r in rr if r["record_type"] == "scheduled_budget"}),
                      "drafted": sum(r["executed_draft_token_count"] for r in ff),
                      "accepted": sum(r["accepted_draft_token_count"] for r in ff),
                      "feedback_records": len(ff), "completion_aggregate": True})
for name in ("single_request", "concurrent_0", "concurrent_1"):
    record = json.loads((run / f"{name}.json").read_text())
    body = json.loads(record["response"])
    assert record["status"] == 200 and body["choices"][0]["message"]["content"].strip()
    assert sum(r["request_id"].startswith(body["id"] + "-") for r in aggregates) == 1
summary = {"git_sha": sha, "worker_source_files_matched_to_git_blobs": len(worker_hashes),
           "record_counts": dict(collections.Counter(r["record_type"] for r in rows)),
           "plan_to_schedule_matches": joined,
           "scheduled_k": sorted({r["scheduled_scalar_k"] for r in scheduled}),
           "effective_scheduler_running_batches": sorted({r["concurrency_batch_b"] for r in plans.values()}),
           "client_concurrency": {"single": 1, "multi": 2},
           "requests": summaries,
           "trim_examples": [r for r in scheduled if r["next_request_placeholder_budget"] < r["scheduled_scalar_k"]][:3],
           "cleanup_evidence": "Three completion aggregates, emitted after adapter.end_request deletes request sessions/history; no later per-request steps. Exact committed source traced, no live heap inspection.",
           "async_feedback_source_lineage": "UNVERIFIED", "tree_execution_verified": False,
           "verdict": "L3 SCALAR PATH VERIFIED"}
(run / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
