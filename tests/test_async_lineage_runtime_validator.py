"""Synthetic validator tests; these are never runtime integration evidence."""
import json

import pytest

from experiments.scalar_pinned_sources import manifest
from experiments.validate_async_lineage_runtime import validate, EXECUTOR, SCHEDULER

SHA = "a" * 40


@pytest.fixture
def evidence(tmp_path):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    save("source-gate.json", dict(git_sha=SHA, vllm="0.30.0", source_gate="PASS",
                                 files=[dict(r, status="PASS") for r in manifest()["files"]]))
    save("launch.json", dict(git_sha=SHA, env={}))
    save("server_env.json", dict(source_commit=SHA))
    save("readiness.json", dict(status="PASS"))
    save("response.json", dict(id="chatcmpl-test", metrics=dict(speculative_decoding=dict(
        num_spec_steps=1, num_draft_tokens=2, num_accepted_draft_tokens=1))))
    save("server-process.json", dict(pid=110))
    common = dict(pid=111, attempt_id="synthetic-attempt", generation_id="synthetic-generation", ledger_id="synthetic-ledger")
    rows = [dict(pid=111, event="hook_installed", vllm_version="0.30.0"),
            dict(pid=111, ppid=110, start_method="spawn", event="enginecore_wrapper_entered", executor_class=EXECUTOR),
            dict(common, event="attempt_created"),
            dict(common, event="executor_published", executor_class=EXECUTOR),
            dict(common, event="scheduler_consumed", consumed=True, same_active_attempt=True),
            dict(common, event="scheduler_initialized", scheduler_class=SCHEDULER),
            dict(common, event="attempt_closed", succeeded=True, context_reset=True, registration_retained=True, consumed=True)]
    (tmp_path / "startup.jsonl").write_text("\n".join(map(json.dumps, rows)))
    rid = "chatcmpl-test-0"
    steps = [dict(record_type="planned_decision", request_id=rid, plan_sequence=2, scalar_runtime_budget=2, batch_scalar_budget=2),
             dict(record_type="scheduled_budget", request_id=rid, schedule_sequence=3, plan_sequence=2, scheduled_scalar_k=2, next_request_placeholder_budget=2),
             dict(record_type="verification_feedback", request_id=rid, source_plan_sequence=2, source_schedule_sequence=3,
                  executed_draft_token_count=2, accepted_draft_token_count=1, lineage_status="attributed",
                  source_request_scalar_k=2, source_batch_scalar_k=2, alpha_ema_before=0.8, alpha_ema=0.74)]
    (tmp_path / "steps.jsonl").write_text("\n".join(map(json.dumps, steps)))
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(dict(tree_execution_verified=False,
        request_id=rid, alpha_ema=0.74, step_count=1, scheduled_runtime_mode="linear_dynamic_k")))
    return tmp_path


def test_complete_synthetic_fixture_passes(evidence):
    assert validate(evidence, SHA)["offline_validator"] == "PASS"


@pytest.mark.parametrize("file,index,field,value", [
    ("startup.jsonl", 0, "pid", 999),
    ("startup.jsonl", 1, "ppid", 999),
    ("startup.jsonl", 4, "attempt_id", "wrong"),
    ("startup.jsonl", 4, "generation_id", "wrong"),
    ("startup.jsonl", 6, "registration_retained", False),
    ("steps.jsonl", 2, "source_schedule_sequence", 99),
    ("steps.jsonl", 2, "source_plan_sequence", None),
    ("steps.jsonl", 2, "executed_draft_token_count", 0),
    ("steps.jsonl", 2, "accepted_draft_token_count", 3),
    ("steps.jsonl", 2, "alpha_ema", 0.8),
    ("steps.jsonl", 2, "tree_execution_verified", True),
    ("steps.jsonl", 2, "profitability", True),
])
def test_tampered_evidence_fails(evidence, file, index, field, value):
    path = evidence / file
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[index][field] = value
    path.write_text("\n".join(map(json.dumps, rows)))
    assert validate(evidence, SHA)["offline_validator"] == "FAIL"


def test_missing_runtime_evidence_fails(tmp_path):
    result = validate(tmp_path, SHA)
    assert result["offline_validator"] == "FAIL"
    assert all(v == "FAIL" for v in result["gates"].values())


def test_wrong_sha_fails(evidence):
    assert validate(evidence, "b" * 40)["offline_validator"] == "FAIL"


def test_source_mismatch_fails(evidence):
    path = evidence / "source-gate.json"
    data = json.loads(path.read_text())
    data["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    assert validate(evidence, SHA)["gates"]["PINNED_SOURCE_GATE"] == "FAIL"
