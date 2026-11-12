"""Fail-closed offline validator for a single canonical async-lineage request."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from experiments.scalar_pinned_sources import manifest

EXECUTOR = "src.server.lineage_executor.LineageUniProcExecutor"
SCHEDULER = "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"
TESTED_SHA = "fb4544166c428e73b164756a2ef0e11ad6b97d6f"
# SHA-256 of the committed, byte-preserved seal at audited HEAD d023e3f.
# Pin the index itself: changing artifacts AND regenerating the index must fail.
SEAL_SHA256 = "8f41e7606244436e2ee6f6aa4b4c9ba4bd1fd484871ea8a180b652117d33bb60"


def validate(root, expected_sha):
    root = Path(root)
    gates, errors = {}, {}
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    def lines(name):
        return [json.loads(s) for s in (root / name).read_text(encoding="utf-8").splitlines() if s]
    def check(name, function):
        try:
            if not function():
                raise ValueError("required evidence absent or inconsistent")
            gates[name] = "PASS"
        except Exception as error:
            gates[name] = "FAIL"
            errors[name] = str(error)

    def source():
        actual = read("source-gate.json")
        expected = {r["path"]: r["sha256"] for r in manifest()["files"]}
        observed = {r["path"]: r["sha256"] for r in actual["files"] if r["status"] == "PASS"}
        return (actual["git_sha"] == expected_sha and actual["vllm"] == "0.30.0"
                and actual["source_gate"] == "PASS" and observed == expected
                and len(actual["files"]) == len(expected)
                and all(r["status"] == "PASS" for r in actual["files"]))
    check("PINNED_SOURCE_GATE", source)
    check("GIT_SHA", lambda: expected_sha == TESTED_SHA and read("launch.json")["git_sha"] == expected_sha
          and read("server_env.json")["source_commit"] == expected_sha)
    def integrity():
        seal = root.parent / "SHA256SUMS.json"
        if hashlib.sha256(seal.read_bytes()).hexdigest() != SEAL_SHA256:
            return False
        entries = json.loads(seal.read_text(encoding="utf-8"))
        return len(entries) == 37 and all(
            hashlib.sha256((root.parent / name).read_bytes()).hexdigest() == digest
            for name, digest in entries.items())
    check("SEALED_EVIDENCE_INTEGRITY", integrity)
    check("SERVER_STARTUP", lambda: read("readiness.json")["status"] == "PASS")

    def bootstrap():
        rows = lines("startup.jsonl")
        wrappers = [r for r in rows if r["event"] == "enginecore_wrapper_entered"]
        if len(wrappers) != 1:
            return False
        wrapper = wrappers[0]
        process_rows = [r for r in rows if r["pid"] == wrapper["pid"]]
        names = [r["event"] for r in process_rows]
        required = ["hook_installed", "enginecore_wrapper_entered", "attempt_created",
                    "executor_published", "scheduler_consumed", "scheduler_initialized", "attempt_closed"]
        if not all(names.count(n) == 1 for n in required):
            return False
        if [names.index(n) for n in required] != sorted(names.index(n) for n in required):
            return False
        values = {r["event"]: r for r in process_rows}
        times = [values[n]["timestamp"] for n in required]
        if not all(math.isfinite(t) for t in times) or not all(a < b for a, b in zip(times, times[1:])):
            return False
        server_pid = read("server-process.json")["pid"]
        return (values["hook_installed"]["vllm_version"] == "0.30.0"
                and wrapper["executor_class"] == EXECUTOR
                and wrapper["pid"] != server_pid and wrapper["ppid"] == server_pid
                and wrapper["start_method"] in ("fork", "spawn", "forkserver"))
    check("HOOK_BOOTSTRAP", bootstrap)

    def attempt():
        rows = lines("startup.jsonl")
        names = ("attempt_created", "executor_published", "scheduler_consumed", "scheduler_initialized", "attempt_closed")
        relevant = [r for r in rows if r["event"] in names]
        selected = {r["event"]: r for r in relevant}
        if len(selected) != 5 or len(relevant) != 5:
            return False
        values = list(selected.values())
        for field in ("pid", "attempt_id"):
            if any(not r.get(field) for r in values) or len({r[field] for r in values}) != 1:
                return False
        for field in ("generation_id", "ledger_id"):
            bound = [r for name, r in selected.items() if name != "attempt_created"]
            if any(not r.get(field) for r in bound) or len({r[field] for r in bound}) != 1:
                return False
        consumed, closed = selected["scheduler_consumed"], selected["attempt_closed"]
        return (consumed["consumed"] and consumed["same_active_attempt"]
                and all(closed[k] for k in ("succeeded", "context_reset", "registration_retained", "consumed"))
                and selected["executor_published"]["executor_class"] == EXECUTOR
                and selected["scheduler_initialized"]["scheduler_class"] == SCHEDULER)
    check("LINEAGE_STARTUP_ATTEMPT", attempt)

    def unique(kind, field):
        rows = [r for r in lines("steps.jsonl") if r["record_type"] == kind]
        keys = [(r["request_id"], r[field]) for r in rows]
        return bool(keys) and len(keys) == len(set(keys)) and all(
            type(r[field]) is int and r[field] > 0 for r in rows)
    check("RAW_PLAN_UNIQUENESS", lambda: unique("planned_decision", "plan_sequence"))
    check("RAW_SCHEDULE_UNIQUENESS", lambda: unique("scheduled_budget", "schedule_sequence"))

    def producers(temporal=False):
        rows = lines("steps.jsonl")
        plans = {(r["request_id"], r["plan_sequence"]): (i, r) for i, r in enumerate(rows)
                 if r["record_type"] == "planned_decision"}
        schedules = {(r["request_id"], r["schedule_sequence"]): (i, r) for i, r in enumerate(rows)
                     if r["record_type"] == "scheduled_budget"}
        feedbacks = [(i, r) for i, r in enumerate(rows) if r["record_type"] == "verification_feedback"]
        if not feedbacks:
            return False
        # This validator is for the sealed single-request, non-preempted positive
        # trace, not a general multiworker algorithm. Positive schedules form an
        # ordered producer stream; feedback consumes its prefix, never the latest
        # compatible plan. Trailing unverified producers at request finish are OK.
        pending = [(i, r) for i, r in enumerate(rows) if r["record_type"] == "scheduled_budget"
                   and r["next_request_placeholder_budget"] > 0]
        response_id = read("response.json")["id"]
        identities = {r["request_id"] for r in rows}
        if len(identities) != 1 or not all(rid == response_id or rid.startswith(response_id + "-") for rid in identities):
            return False
        for ordinal, (fi, feedback) in enumerate(feedbacks):
            rid = feedback["request_id"]
            pi, plan = plans[(rid, feedback["source_plan_sequence"])]
            si, schedule = schedules[(rid, feedback["source_schedule_sequence"])]
            if (schedule["plan_sequence"] != plan["plan_sequence"]
                    or plan["scalar_runtime_budget"] != feedback["source_request_scalar_k"]
                    or plan["batch_scalar_budget"] != feedback["source_batch_scalar_k"]
                    or schedule["scheduled_scalar_k"] != feedback["source_batch_scalar_k"]
                    or schedule["next_request_placeholder_budget"] != feedback["source_request_scalar_k"]):
                return False
            if temporal:
                times = [plan["recorded_at"], schedule["recorded_at"], feedback["recorded_at"]]
                if (not pi < si < fi or not all(math.isfinite(t) for t in times)
                        or not times[0] < times[1] < times[2] or pending[ordinal][0] != si):
                    return False
        return True
    check("SOURCE_PRODUCER_JOIN", producers)
    check("TEMPORAL_PRODUCER", lambda: producers(temporal=True))

    def log_scan():
        # Scan emitted runtime artifacts only, never explanatory README/worklog.
        pattern = re.compile(r"missing[- ]lineage|stale[- ]lineage|duplicate[- ]lineage|lineage[- ]mismatch|"
                             r"feedback[- ]skipped|provenance[- ]failure|stale-or-mismatched-lineage|"
                             r"lineage registration failed|feedback failed|scheduled-budget telemetry failed|"
                             r"planning failed|telemetry failed", re.I)
        paths = [root / "vllm.log", root.parent / "modal-integration-04.log",
                 root / "steps.jsonl", root / "telemetry.jsonl", root / "startup.jsonl"]
        return all(not pattern.search(path.read_text(encoding="utf-8")) for path in paths)
    check("LINEAGE_FAILURE_LOG_SCAN", log_scan)

    def feedback():
        rows = lines("steps.jsonl")
        feedbacks = [r for r in rows if r["record_type"] == "verification_feedback"]
        if not feedbacks or any(r["record_type"] == "verification_feedback_skipped" for r in rows):
            return False
        response_id = read("response.json")["id"]
        planned = {(r["request_id"], r["plan_sequence"]): r for r in rows if r["record_type"] == "planned_decision"}
        scheduled = {(r["request_id"], r["schedule_sequence"]): r for r in rows if r["record_type"] == "scheduled_budget"}
        nonnull = 0
        previous_alpha = {}
        for row in feedbacks:
            rid = row["request_id"]
            if not (rid == response_id or rid.startswith(response_id + "-")):
                return False
            drafted, accepted = row["executed_draft_token_count"], row["accepted_draft_token_count"]
            if not (type(drafted) is int and type(accepted) is int and 0 <= accepted <= drafted and drafted > 0):
                return False
            schedule = scheduled[(rid, row["source_schedule_sequence"])]
            if (row["lineage_status"] != "attributed" or schedule["plan_sequence"] != row["source_plan_sequence"]
                    or schedule["scheduled_scalar_k"] != row["source_batch_scalar_k"]
                    or schedule["next_request_placeholder_budget"] != row["source_request_scalar_k"]
                    or drafted > row["source_request_scalar_k"]):
                return False
            if row["source_plan_sequence"] is not None:
                plan = planned[(rid, row["source_plan_sequence"])]
                if (plan["scalar_runtime_budget"] != row["source_request_scalar_k"]
                        or plan["batch_scalar_budget"] != row["source_batch_scalar_k"]):
                    return False
                nonnull += 1
            before, after = row["alpha_ema_before"], row["alpha_ema"]
            # Canonical controller kwargs are empty; default ema_beta is 0.8.
            if not math.isclose(after, 0.8 * before + (1.0 - 0.8) * (accepted / drafted), rel_tol=0, abs_tol=1e-12):
                return False
            if rid in previous_alpha and not math.isclose(before, previous_alpha[rid], rel_tol=0, abs_tol=1e-12):
                return False
            previous_alpha[rid] = after
        aggregates = lines("telemetry.jsonl")
        if len(aggregates) != 1:
            return False
        aggregate = aggregates[0]
        stats = read("response.json")["metrics"]["speculative_decoding"]
        return (nonnull == len(feedbacks) == 13 and len(previous_alpha) == 1
                and math.isclose(feedbacks[0]["alpha_ema_before"], 0.8, rel_tol=0, abs_tol=1e-12)
                and read("launch.json")["env"].get("ADAPTIVE_CONTROLLER_KWARGS", "") == ""
                and aggregate["request_id"] in previous_alpha
                and aggregate["step_count"] == len(feedbacks) == stats["num_spec_steps"]
                and aggregate["scheduled_runtime_mode"] == "linear_dynamic_k"
                and math.isclose(aggregate["alpha_ema"], previous_alpha[aggregate["request_id"]], rel_tol=0, abs_tol=1e-12)
                and stats["num_draft_tokens"] == sum(r["executed_draft_token_count"] for r in feedbacks) == 49
                and stats["num_accepted_draft_tokens"] == sum(r["accepted_draft_token_count"] for r in feedbacks) == 49)
    check("REAL_SPECULATIVE_FEEDBACK", feedback)

    def scope():
        rows = lines("steps.jsonl") + lines("telemetry.jsonl")
        forbidden = {"speedup_pct", "throughput_improvement", "latency_improvement", "profitability", "tpot_improvement"}
        def valid(value):
            if isinstance(value, dict):
                return (value.get("tree_execution_verified") is not True
                        and not forbidden.intersection(value) and all(valid(v) for v in value.values()))
            return all(valid(v) for v in value) if isinstance(value, list) else True
        return bool(rows) and valid(rows)
    check("SCOPE_CEILING", scope)
    return dict(offline_validator="PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
                git_sha=expected_sha, gates=gates, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--no-write", action="store_true", help="Print results without modifying evidence")
    args = parser.parse_args()
    result = validate(args.root, args.expected_sha)
    if not args.no_write:
        seal = args.root.parent / "SHA256SUMS.json"
        if seal.exists():
            parser.error("sealed evidence output is protected; use --no-write")
        (args.root / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["offline_validator"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
