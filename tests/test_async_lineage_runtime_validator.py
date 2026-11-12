"""Adversarial tests on temporary mirrors; never mutate sealed runtime evidence."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.validate_async_lineage_runtime import validate

SHA = "fb4544166c428e73b164756a2ef0e11ad6b97d6f"
CANONICAL = Path(__file__).resolve().parents[1] / "experiments/runs/m4_02_04/20261112T014441Z-async-lineage-pinned-runtime"


def sealed_hashes():
    return {name: hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest()
            for name in json.loads((CANONICAL / "SHA256SUMS.json").read_text())}


@pytest.fixture(scope="module", autouse=True)
def preserve_originals():
    before = sealed_hashes()
    assert before == json.loads((CANONICAL / "SHA256SUMS.json").read_text())
    yield
    assert sealed_hashes() == before


@pytest.fixture
def evidence(tmp_path):
    shutil.copytree(CANONICAL, tmp_path / "mirror")
    return tmp_path / "mirror/runtime-04"


def edit(root, file, mutate):
    path = root / file
    if file.endswith("jsonl"):
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        mutate(rows)
        path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    else:
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")


def record(rows, kind):
    return next(r for r in rows if r.get("record_type", r.get("event")) == kind)


def test_canonical_mirror_passes(evidence):
    assert validate(evidence, SHA)["offline_validator"] == "PASS"


@pytest.mark.parametrize("case,gate", [
    ("duplicate_plan", "RAW_PLAN_UNIQUENESS"),
    ("duplicate_schedule", "RAW_SCHEDULE_UNIQUENESS"),
    ("missing_plan", "SOURCE_PRODUCER_JOIN"),
    ("missing_schedule", "SOURCE_PRODUCER_JOIN"),
    ("request_mismatch", "SOURCE_PRODUCER_JOIN"),
    ("latest_plan_redirection", "TEMPORAL_PRODUCER"),
    ("future_producer", "TEMPORAL_PRODUCER"),
    ("accepted_over_drafted", "REAL_SPECULATIVE_FEEDBACK"),
    ("zero_drafted", "REAL_SPECULATIVE_FEEDBACK"),
    ("negative_accepted", "REAL_SPECULATIVE_FEEDBACK"),
    ("boolean_count", "REAL_SPECULATIVE_FEEDBACK"),
])
def test_raw_adversaries(evidence, case, gate):
    def mutate(rows):
        plan = record(rows, "planned_decision")
        schedule = next(r for r in rows if r["record_type"] == "scheduled_budget" and r["plan_sequence"] == 1)
        feedback = record(rows, "verification_feedback")
        if case.startswith("duplicate_"):
            rows.append(dict(plan if case == "duplicate_plan" else schedule))
        elif case == "missing_plan":
            rows.remove(plan)
        elif case == "missing_schedule":
            rows.remove(schedule)
        elif case == "request_mismatch":
            plan["request_id"] = "wrong-request"
        elif case == "latest_plan_redirection":
            # Plan 3 / schedule 4 are already logged and have compatible K=3.
            feedback.update(source_plan_sequence=3, source_schedule_sequence=4)
        elif case == "future_producer":
            plan["recorded_at"] = feedback["recorded_at"] + 1
        else:
            feedback.update({"accepted_draft_token_count": 4} if case == "accepted_over_drafted" else
                            {"executed_draft_token_count": 0} if case == "zero_drafted" else
                            {"accepted_draft_token_count": -1} if case == "negative_accepted" else
                            {"executed_draft_token_count": True})
    edit(evidence, "steps.jsonl", mutate)
    result = validate(evidence, SHA)
    assert result["offline_validator"] == "FAIL"
    # Prove semantic rejection independently of the seal mismatch.
    assert result["gates"][gate] == "FAIL", result


@pytest.mark.parametrize("file,field,value,gate", [
    ("telemetry.jsonl", "step_count", 12, "REAL_SPECULATIVE_FEEDBACK"),
    ("telemetry.jsonl", "alpha_ema", 0.9, "REAL_SPECULATIVE_FEEDBACK"),
    ("telemetry.jsonl", "request_id", "wrong", "REAL_SPECULATIVE_FEEDBACK"),
    ("response.json", "num_draft_tokens", 50, "REAL_SPECULATIVE_FEEDBACK"),
    ("response.json", "num_accepted_draft_tokens", 50, "REAL_SPECULATIVE_FEEDBACK"),
    ("launch.json", "git_sha", "a" * 40, "GIT_SHA"),
    ("source-gate.json", "sha256", "0" * 64, "PINNED_SOURCE_GATE"),
])
def test_aggregate_and_attribution_adversaries(evidence, file, field, value, gate):
    def mutate(data):
        target = data[0] if isinstance(data, list) else data
        if file == "response.json":
            target = data["metrics"]["speculative_decoding"]
        elif file == "source-gate.json":
            target = data["files"][0]
        target[field] = value
    edit(evidence, file, mutate)
    assert validate(evidence, SHA)["gates"][gate] == "FAIL"


@pytest.mark.parametrize("case", ["missing_hook", "late_hook", "attempt_id", "generation_id", "ledger_id"])
def test_bootstrap_adversaries(evidence, case):
    def mutate(rows):
        if case == "missing_hook":
            rows.remove(next(r for r in rows if r["event"] == "hook_installed" and r["pid"] == 32))
        elif case == "late_hook":
            next(r for r in rows if r["event"] == "hook_installed" and r["pid"] == 32)["timestamp"] = 2e9
        else:
            record(rows, "scheduler_consumed")[case] = "wrong"
    edit(evidence, "startup.jsonl", mutate)
    gate = "HOOK_BOOTSTRAP" if case in ("missing_hook", "late_hook") else "LINEAGE_STARTUP_ATTEMPT"
    assert validate(evidence, SHA)["gates"][gate] == "FAIL"


@pytest.mark.parametrize("warning", ["missing-lineage", "stale lineage", "duplicate lineage", "lineage mismatch",
                                     "feedback skipped", "provenance failure", "stale-or-mismatched-lineage",
                                     "lineage registration failed", "feedback failed"])
def test_log_only_failure(evidence, warning):
    with (evidence / "vllm.log").open("a", encoding="utf-8") as stream:
        stream.write(f"\nWARNING Adaptive {warning} for chatcmpl-863a80d9a3f74fb9-b0204581\n")
    assert validate(evidence, SHA)["gates"]["LINEAGE_FAILURE_LOG_SCAN"] == "FAIL"


def test_sealed_byte_corruption(evidence):
    with (evidence / "telemetry.jsonl").open("ab") as stream:
        stream.write(b" ")
    assert validate(evidence, SHA)["gates"]["SEALED_EVIDENCE_INTEGRITY"] == "FAIL"


def test_resealed_corruption_rejected(evidence):
    path = evidence.parent / "SHA256SUMS.json"
    data = json.loads(path.read_text())
    data["runtime-04/telemetry.jsonl"] = "0" * 64
    path.write_text(json.dumps(data))
    assert validate(evidence, SHA)["gates"]["SEALED_EVIDENCE_INTEGRITY"] == "FAIL"


def test_wrong_expected_sha(evidence):
    assert validate(evidence, "a" * 40)["gates"]["GIT_SHA"] == "FAIL"


def test_missing_evidence(tmp_path):
    result = validate(tmp_path, SHA)
    assert result["offline_validator"] == "FAIL"
    assert all(v == "FAIL" for v in result["gates"].values())


def test_cli_read_only(evidence):
    before = {p: p.read_bytes() for p in evidence.parent.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", "-m", "experiments.validate_async_lineage_runtime",
                             str(evidence), "--expected-sha", SHA, "--no-write"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(p.read_bytes() == content for p, content in before.items())


def test_cli_refuses_sealed_output(evidence):
    path = evidence / "validation.json"
    before = path.read_bytes()
    result = subprocess.run([sys.executable, "-B", "-m", "experiments.validate_async_lineage_runtime",
                             str(evidence), "--expected-sha", SHA], capture_output=True, text=True)
    assert result.returncode != 0
    assert path.read_bytes() == before

@pytest.mark.parametrize("file,event,field,value,gate", [
    ("startup.jsonl", "enginecore_wrapper_entered", "ppid", 999, "HOOK_BOOTSTRAP"),
    ("startup.jsonl", "enginecore_wrapper_entered", "executor_class", "wrong", "HOOK_BOOTSTRAP"),
    ("startup.jsonl", "attempt_closed", "registration_retained", False, "LINEAGE_STARTUP_ATTEMPT"),
    ("startup.jsonl", "scheduler_initialized", "scheduler_class", "wrong", "LINEAGE_STARTUP_ATTEMPT"),
    ("steps.jsonl", "verification_feedback", "source_plan_sequence", None, "SOURCE_PRODUCER_JOIN"),
    ("steps.jsonl", "verification_feedback", "source_schedule_sequence", 999, "SOURCE_PRODUCER_JOIN"),
    ("steps.jsonl", "verification_feedback", "alpha_ema", 0.8, "REAL_SPECULATIVE_FEEDBACK"),
    ("steps.jsonl", "verification_feedback", "alpha_ema", 0.8400000001, "REAL_SPECULATIVE_FEEDBACK"),
    ("steps.jsonl", "verification_feedback", "lineage_status", "wrong", "REAL_SPECULATIVE_FEEDBACK"),
    ("steps.jsonl", "verification_feedback", "tree_execution_verified", True, "SCOPE_CEILING"),
    ("steps.jsonl", "verification_feedback", "profitability", True, "SCOPE_CEILING"),
])
def test_retained_contracts(evidence, file, event, field, value, gate):
    edit(evidence, file, lambda rows: record(rows, event).update({field: value}))
    assert validate(evidence, SHA)["gates"][gate] == "FAIL"


def test_feedback_skipped(evidence):
    edit(evidence, "steps.jsonl", lambda rows: rows.append(dict(record_type="verification_feedback_skipped")))
    assert validate(evidence, SHA)["gates"]["REAL_SPECULATIVE_FEEDBACK"] == "FAIL"
