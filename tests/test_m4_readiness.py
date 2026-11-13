"""SYNTHETIC raw diagnostic fixtures test evaluator only; never GPU evidence."""
import json
from copy import deepcopy

import pytest

from experiments.m4_a100_profile import build_profile, validate_profile_file, BATCHES, KS
from experiments.m4_protocol import sha256, write_new
from experiments.m4_readiness import evaluate
from experiments.m4_treatments import TREATMENTS, definition, TARGET_REVISION
from experiments.m4_cost_profiler import TIMING
from test_m4_a100_profile import synthetic_profile, SHA
from test_m4_lossless import fixtures


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def diagnostic_fixture(tmp_path):
    profile_dir = tmp_path / "profile_inputs"; profile_dir.mkdir()
    env = dict(synthetic_profile()["metadata"], canonical_hardware=True, expected_gpu="A100",
               run_id="SYNTHETIC", run_type="component_benchmark", working_tree_clean=True,
               target_revision=TARGET_REVISION, draft_revision=definition("eagle_fixed")["speculative_config"]["revision"],
               contexts=[1024], batches=list(BATCHES), batch_grid=list(BATCHES),
               k_values=list(KS), k_grid=list(KS), warmup=5, repeats=50, timing_method=TIMING)
    records = [dict(run_id=env["run_id"], source_commit=SHA, gpu=env["gpu"],
                    target_revision=env["target_revision"], draft_revision=env["draft_revision"],
                    timing_method=TIMING, measurement_kind=kind, batch_size=b, K=k, context_len=1024,
                    repeat=i, is_warmup=warm, status="ok", latency_ms=1.0,
                    observed_shapes=[dict(actual_batch_size=b, query_tokens_per_request=[k+1]*b,
                                          context_lengths=[1024]*b, preempted_req_ids=[])])
               for b in BATCHES for k in KS
               for kind in (("linear_verify_forward", "eagle_propose") if k else ("target_decode_forward",))
               for warm, count in ((True, 5), (False, 50)) for i in range(count)]
    jsonl(profile_dir / "raw.jsonl", records)
    write_new(profile_dir / "env.json", env)
    p = build_profile(env, records, sha256((profile_dir / "raw.jsonl").read_bytes()))
    write_new(profile_dir / "profile_sm80.json", p)
    write_new(tmp_path / "profile_sm80.json", p)
    write_new(tmp_path / "environment.json", dict(env, execution_backend="google_colab"))
    expected, captures = fixtures()
    write_new(tmp_path / "diagnostic.json", dict(expected_prompts=expected, profile_sha256=sha256((tmp_path / "profile_sm80.json").read_bytes())))
    jsonl(tmp_path / "captures.jsonl", captures)
    for t in TREATMENTS:
        (tmp_path / t).mkdir()
        write_new(tmp_path / t / "launch.json", dict(definition=definition(t), source_commit=SHA))
        write_new(tmp_path / t / "lifecycle.json", dict(gpu_before={"memory": "0,40000", "processes": ""},
                                                     gpu_after={"memory": "0,40000", "processes": ""}))
        (tmp_path / t / "server.log").write_text("SYNTHETIC successful log\n")
        jsonl(tmp_path / t / "responses.jsonl", [dict(metrics=dict(speculative_decoding=dict(
              num_draft_tokens=16, num_accepted_draft_tokens=16, num_spec_steps=5)))])
    steps, events = [], [dict(event="scheduler_initialized", scheduler_class="src.server.adaptive_scheduler.AdaptiveAsyncScheduler")]
    alpha = .8
    for seq in range(1, 7):
        k = 3 if alpha < .9 else 4
        steps.extend([dict(record_type="planned_decision", request_id="r", plan_sequence=seq,
            controller_class="src.proposer.controller.AdaptiveDraftController", scalar_runtime_budget=k,
            batch_scalar_budget=k, acceptance_rate_ema=alpha, recorded_at=seq*3),
            dict(record_type="scheduled_budget", request_id="r", schedule_sequence=seq,
                 plan_sequence=seq, scheduled_scalar_k=k)])
        if seq < 6:
            events.append(dict(event="draft_produced", batch_k=k, rows=[dict(request_id="r", schedule_sequence=seq,
                               plan_sequence=seq, request_k=k, batch_k=k)]))
            after = .8 * alpha + .2
            steps.append(dict(record_type="verification_feedback", request_id="r", source_schedule_sequence=seq,
                source_plan_sequence=seq, executed_draft_token_count=k, accepted_draft_token_count=k,
                lineage_status="attributed", alpha_ema_before=alpha, alpha_ema=after, recorded_at=seq*3+1))
            alpha = after
    events.append(dict(event="request_cleanup", request_id="r", active_sessions=0, active_lineage=0))
    jsonl(tmp_path / "eagle_adaptive/steps.jsonl", steps)
    jsonl(tmp_path / "eagle_adaptive/lineage.jsonl", events)
    jsonl(tmp_path / "eagle_adaptive/requests.jsonl", [dict(request_id="r")])
    return tmp_path


def test_all_required_raw_contracts_pass_synthetic_fixture(diagnostic_fixture):
    report = evaluate(diagnostic_fixture, SHA)
    assert report["ADAPTIVE_READY"] == "PASS", report["errors"]


@pytest.mark.parametrize("filename,change", [
    ("environment.json", {"gpu": "NVIDIA L4"}),
    ("environment.json", {"source_commit": "b" * 40}),
    ("environment.json", {"vllm_version": "latest"}),
    ("profile_sm80.json", {"models": {}}),
    ("eagle_adaptive/launch.json", {"definition": {}}),
    ("eagle_adaptive/lifecycle.json", {"gpu_after": {"memory": "0,40000", "processes": "99,1000"}}),
])
def test_single_failed_contract_blocks(diagnostic_fixture, filename, change):
    path = diagnostic_fixture / filename
    data = json.loads(path.read_text()); data.update(change); path.write_text(json.dumps(data))
    assert evaluate(diagnostic_fixture, SHA)["ADAPTIVE_READY"] == "FAIL"


@pytest.mark.parametrize("failure", ["constant_k", "wrong_lineage", "no_feedback", "wrong_accept", "no_cleanup", "fallback", "lossless"])
def test_runtime_evidence_rejections(diagnostic_fixture, failure):
    root = diagnostic_fixture
    path = root / "eagle_adaptive/steps.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "constant_k":
        for r in steps:
            if r["record_type"] == "planned_decision": r["scalar_runtime_budget"] = 3
    if failure == "wrong_lineage":
        next(r for r in steps if r["record_type"] == "verification_feedback")["source_schedule_sequence"] = 999
    if failure == "no_feedback": steps = [r for r in steps if r["record_type"] != "verification_feedback"]
    if failure == "wrong_accept":
        next(r for r in steps if r["record_type"] == "verification_feedback")["accepted_draft_token_count"] = 100
    jsonl(path, steps)
    if failure == "no_cleanup": (root / "eagle_adaptive/requests.jsonl").write_text("")
    if failure == "fallback": (root / "eagle_fixed/server.log").write_text("falling back to vanilla")
    if failure == "lossless":
        rows = [json.loads(line) for line in (root / "captures.jsonl").read_text().splitlines()]
        rows[-1]["token_ids"] = [9]; jsonl(root / "captures.jsonl", rows)
    assert evaluate(root, SHA)["ADAPTIVE_READY"] == "FAIL"


def test_measured_label_without_raw_not_enough(diagnostic_fixture):
    path = diagnostic_fixture / "profile_inputs/raw.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError): validate_profile_file(path.parent / "profile_sm80.json", SHA)
    assert evaluate(diagnostic_fixture, SHA)["ADAPTIVE_READY"] == "FAIL"
