"""Fail-closed evaluator of raw A100 diagnostics, never a local runtime claim."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from experiments.m4_a100_profile import hardware_gate, validate_profile_file
from experiments.m4_lossless import compare_captures
from experiments.m4_protocol import sha256
from experiments.m4_runtime import require_gpu_release
from experiments.m4_treatments import TREATMENTS, definition, SCHEDULER


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(directory: Path, source_commit: str) -> dict:
    checks, errors = {}, {}

    def check(name, operation):
        try:
            if operation() is False:
                raise ValueError("Required evidence does not satisfy contract")
            checks[name] = "PASS"
        except Exception as exc:
            checks[name] = "FAIL"
            errors[name] = f"{type(exc).__name__}: {exc}"

    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    def environment():
        env = read("environment.json")
        hardware_gate(env)
        return (bool(re.fullmatch(r"[0-9a-f]{40}", source_commit))
                and env["source_commit"] == source_commit and env["execution_backend"] == "google_colab")
    check("A100_SOURCE_ENVIRONMENT", environment)

    def profile():
        p = read("profile_sm80.json")
        validate_profile_file(directory / "profile_inputs/profile_sm80.json", source_commit)
        if p != read("profile_inputs/profile_sm80.json"):
            return False
        env = read("environment.json")
        if any(p["metadata"].get(field) != env.get(field) for field in
               ("gpu", "gpu_compute_capability", "gpu_memory_total_mb", "driver_version",
                "torch_version", "cuda_version", "vllm_version")):
            return False
        return sha256((directory / "profile_sm80.json").read_bytes()) == read("diagnostic.json")["profile_sha256"]
    check("MEASURED_PROFILE", profile)

    def identities():
        for treatment in TREATMENTS:
            launch = read(f"{treatment}/launch.json")
            if launch["definition"] != definition(treatment) or launch["source_commit"] != source_commit:
                return False
        return True
    check("TREATMENT_IDENTITIES", identities)

    def lossless():
        report = compare_captures(read_lines(directory / "captures.jsonl"),
                                  read("diagnostic.json")["expected_prompts"], source_commit)
        return report["passed"]
    check("GREEDY_LOSSLESS_AND_REPEATABILITY", lossless)

    def metrics():
        for treatment in TREATMENTS[1:]:
            responses = read_lines(directory / treatment / "responses.jsonl")
            counts = [r["metrics"]["speculative_decoding"] for r in responses]
            if not counts:
                return False
            for c in counts:
                d, a, steps = (c[f] for f in ("num_draft_tokens", "num_accepted_draft_tokens", "num_spec_steps"))
                if not all(type(n) is int for n in (d, a, steps)) or not 0 <= a <= d or d <= 0 or steps <= 0:
                    return False
        return True
    check("REAL_NATIVE_DRAFT_AND_ACCEPT_METRICS", metrics)

    def controller():
        steps = read_lines(directory / "eagle_adaptive/steps.jsonl")
        startup = read_lines(directory / "eagle_adaptive/lineage.jsonl")
        if not any(r.get("scheduler_class") == SCHEDULER and r["event"] == "scheduler_initialized" for r in startup):
            return False
        plans = {(r["request_id"], r["plan_sequence"]): r for r in steps if r["record_type"] == "planned_decision"}
        schedules = {(r["request_id"], r["schedule_sequence"]): r for r in steps if r["record_type"] == "scheduled_budget"}
        births = {(row["request_id"], row["schedule_sequence"]): (r, row)
                  for r in startup if r["event"] == "draft_produced" for row in r["rows"]}
        feedbacks = [r for r in steps if r["record_type"] == "verification_feedback"]
        if not feedbacks or any(r["record_type"] == "verification_feedback_skipped" for r in steps):
            return False
        meaningful, updated = set(), False
        for f in feedbacks:
            rid, seq = f["request_id"], f["source_schedule_sequence"]
            scheduled = schedules[(rid, seq)]
            birth, row = births[(rid, seq)]
            # Completed-prefill proposals have no decode plan; never invent one.
            if f["source_plan_sequence"] is None:
                continue
            plan = plans[(rid, f["source_plan_sequence"])]
            if plan.get("controller_class") != "src.proposer.controller.AdaptiveDraftController":
                return False
            drafted, accepted = f["executed_draft_token_count"], f["accepted_draft_token_count"]
            if (f["lineage_status"] != "attributed" or not 0 <= accepted <= drafted <= row["request_k"]
                    or birth["batch_k"] != plan["batch_scalar_budget"]
                    or scheduled["scheduled_scalar_k"] != birth["batch_k"]
                    or row["request_k"] != plan["scalar_runtime_budget"]
                    or f["source_plan_sequence"] != scheduled["plan_sequence"]):
                return False
            before, after = f["alpha_ema_before"], f["alpha_ema"]
            if drafted <= 0 or not math.isclose(after, .8 * before + .2 * accepted / drafted, abs_tol=1e-12):
                return False
            # A later plan must observe the updated state for this same request.
            later = [p for (request, n), p in plans.items() if request == rid
                     and n > f["source_plan_sequence"] and p["recorded_at"] > f["recorded_at"]]
            if later:
                meaningful.add(plan["scalar_runtime_budget"])
                updated |= any(math.isclose(p["acceptance_rate_ema"], after, abs_tol=1e-12) for p in later)
        native = [r["metrics"]["speculative_decoding"]
                  for r in read_lines(directory / "eagle_adaptive/responses.jsonl")]
        if (sum(f["executed_draft_token_count"] for f in feedbacks) != sum(n["num_draft_tokens"] for n in native)
                or sum(f["accepted_draft_token_count"] for f in feedbacks) != sum(n["num_accepted_draft_tokens"] for n in native)):
            return False
        return len(meaningful) >= 2 and updated
    check("PROJECT_CONTROLLER_MULTI_K_EXECUTION_LINEAGE", controller)

    def cleanup():
        for treatment in TREATMENTS:
            lifecycle = read(f"{treatment}/lifecycle.json")
            require_gpu_release(lifecycle["gpu_before"], lifecycle["gpu_after"])
        events = read_lines(directory / "eagle_adaptive/lineage.jsonl")
        clean = [r for r in events if r["event"] == "request_cleanup"]
        aggregates = read_lines(directory / "eagle_adaptive/requests.jsonl")
        return bool(clean and aggregates) and clean[-1]["active_sessions"] == clean[-1]["active_lineage"] == 0
    check("REQUEST_AND_GPU_LIFECYCLE", cleanup)

    def no_fallback():
        needles = ("planning failed", "feedback failed", "lineage registration failed", "provenance unavailable",
                   "adaptive feedback skipped", "falling back", "traceback (most recent", "out of memory")
        return all(not any(s in (directory / t / "server.log").read_text(encoding="utf-8").lower()
                           for s in needles) for t in TREATMENTS)
    check("NO_CRASH_OR_SILENT_FALLBACK", no_fallback)
    return dict(ADAPTIVE_READY="PASS" if all(v == "PASS" for v in checks.values()) else "FAIL",
                source_commit=source_commit, checks=checks, errors=errors,
                evidence_sha256={str(p.relative_to(directory)): sha256(p.read_bytes())
                                 for p in directory.rglob("*") if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = evaluate(args.directory, args.source_commit)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ADAPTIVE_READY"] == "PASS" else 1)


if __name__ == "__main__":
    main()
