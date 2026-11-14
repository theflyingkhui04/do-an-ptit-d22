"""SYNTHETIC campaign records; no serving subprocess is launched."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.m4_protocol import ROOT, build_manifest, validate_manifest, git, write_new
from experiments.run_m4_colab_campaign import (attempt_state, completed_attempt, validate_rerun_assessment,
                                               campaign, cell_status, validate_cell)
from experiments.m4_git_evidence import copy_immutable, artifact_files, git_auth
from experiments import aggregate_results as agg
from test_aggregate_results import make_record, spec_metrics


def test_exact_manifest_enumeration_and_blob_hashes():
    sha = git("rev-parse", "HEAD")
    manifest = build_manifest(sha)
    validate_manifest(manifest, sha)
    assert len(manifest["cells"]) == len({c["cell_id"] for c in manifest["cells"]}) == 54
    assert [d["count"] for d in manifest["workloads"].values()] == [30, 12, 12]
    assert {c["treatment"] for c in manifest["cells"]} == {"vanilla", "eagle_fixed", "eagle_adaptive"}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "revision", "source", "requests"])
def test_manifest_drift_rejected(kind):
    sha = git("rev-parse", "HEAD"); m = build_manifest(sha)
    if kind == "missing": m["cells"].pop()
    if kind == "duplicate": m["cells"][-1] = deepcopy(m["cells"][0])
    if kind == "revision": m["cells"][-1]["speculative_config"]["revision"] = "bad"
    if kind == "source": m["source_commit"] = "b" * 40
    if kind == "requests": m["cells"][0]["measured_requests"] = 1
    with pytest.raises(ValueError): validate_manifest(m, sha)


@pytest.mark.parametrize("state", ["FAILED", "OOM", "TIMEOUT", "INVALID", "RUNNING"])
def test_failed_or_abandoned_attempt_never_silently_reruns(tmp_path, state):
    a = tmp_path / "attempt-old"; a.mkdir()
    write_new(a / "started.json", {"state": "RUNNING"})
    if state != "RUNNING": write_new(a / "finished.json", {"state": state})
    assert attempt_state(a) == state
    with pytest.raises(ValueError, match="assessment"):
        completed_attempt({}, tmp_path, {})
    assessment = dict(prior_attempts=[a.name], previous_failure=state, root_cause_hypothesis="SYNTHETIC",
                      offline_fix_evidence="SYNTHETIC", expected_pass_fail_signal="SYNTHETIC")
    validate_rerun_assessment(tmp_path, assessment)
    assessment.pop("offline_fix_evidence")
    with pytest.raises(ValueError): validate_rerun_assessment(tmp_path, assessment)


def test_no_runtime_work_before_raw_readiness_pass(tmp_path, monkeypatch):
    import experiments.run_m4_colab_campaign as module
    monkeypatch.setattr(module, "environment", lambda _: {})
    monkeypatch.setattr(module, "profile_gate", lambda *_: "hash")
    monkeypatch.setattr(module, "evaluate", lambda *_: {"ADAPTIVE_READY": "FAIL"})
    monkeypatch.setattr(module, "verify_diagnostic_persisted", lambda *_: pytest.fail("Must not launch/publish"))
    with pytest.raises(ValueError, match="prohibited"):
        campaign(SimpleNamespace(source_commit="a" * 40, profile=tmp_path, diagnostic=tmp_path))


def test_immutable_copy_allows_additions_but_not_rewrites(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir()
    (a / "failure.json").write_text('{"status":"failed"}')
    copy_immutable(a, b); copy_immutable(a, b)
    (a / "later.json").write_text('{}'); copy_immutable(a, b)
    (a / "failure.json").write_text('{"status":"pass"}')
    with pytest.raises(ValueError, match="immutable"): copy_immutable(a, b)
    assert json.loads((b / "failure.json").read_text())["status"] == "failed"


def test_secret_and_cache_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "SYNTHETIC-secret-test")
    (tmp_path / "log.txt").write_text("SYNTHETIC-secret-test")
    with pytest.raises(ValueError, match="Secret"): list(artifact_files(tmp_path))


def test_askpass_ephemeral_and_no_token_in_helper(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "SYNTHETIC-test-token")
    with git_auth() as env:
        path = Path(env["GIT_ASKPASS"])
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert "SYNTHETIC-test-token" not in path.read_text()
    assert not path.exists()


@pytest.mark.parametrize("treatment", ["eagle_fixed", "eagle_adaptive"])
def test_new_treatments_aggregate_native_metrics_and_variance(tmp_path, treatment):
    rows = [make_record(lane=treatment, repeat_id=f"r{i:03d}", tpot_ms=20+i,
                        server_metrics=spec_metrics(accepted=2, drafted=3, steps=1)) for i in range(1, 6)]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = agg.aggregate_raw_jsonls([path])[0]
    assert result["alpha"] == pytest.approx(2/3)
    assert result["tau"] == 3
    assert result["n_runs"] == 5 and isinstance(result["variance_flag"], bool)
    rows[-1]["success"] = False
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError): agg.aggregate_raw_jsonls([path])


def test_adaptive_zero_drafts_undefined_alpha(tmp_path):
    record = make_record(lane="eagle_adaptive", server_metrics=spec_metrics(accepted=0, drafted=0, steps=0))
    path = tmp_path / "raw.jsonl"; path.write_text(json.dumps(record))
    assert agg.aggregate_raw_jsonls([path])[0]["alpha"] == "N/A"


def synthetic_valid_cell(tmp_path):
    sha = git("rev-parse", "HEAD"); manifest = build_manifest(sha)
    cell = manifest["cells"][0]
    directory = tmp_path / cell["cell_id"] / "attempt-synthetic"
    directory.mkdir(parents=True)
    write_new(directory / "started.json", dict(source_commit=sha, cell=cell, campaign_id="SYNTHETIC"))
    rows = []
    workload = manifest["workloads"][cell["workload"]]
    for repeat, indices in enumerate(cell["sample_indices_by_repeat"], 1):
        for number, index in enumerate(indices):
            rows.append(dict(make_record(repeat_id=f"r{repeat:03d}"), request_id=f"synthetic-{repeat}-{number}",
                experiment_id="SYNTHETIC", commit=sha, workload_file_sha256=workload["sha256"],
                temperature=0, seed=42, sample_policy="cycle-rotate", sample_index=index,
                sample_id=workload["prompt_ids"][index], max_completion_tokens=workload["request_settings"][index]["max_tokens"]))
    (directory / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_new(directory / "server/launch.json", dict(definition=definition(cell["treatment"]), source_commit=sha))
    (directory / "server/server.log").write_text("SYNTHETIC")
    return manifest, cell, directory


def test_resume_skips_only_revalidated_immutable_completed_cell(tmp_path):
    from experiments.m4_protocol import sha256
    m, c, d = synthetic_valid_cell(tmp_path)
    assert validate_cell(c, d, m)["n_runs"] == 5
    write_new(d / "finished.json", dict(state="COMPLETE_VALID", raw_sha256=sha256((d / "raw.jsonl").read_bytes())))
    assert completed_attempt(c, d.parent, m) == d
    with (d / "raw.jsonl").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        completed_attempt(c, d.parent, m)


@pytest.mark.parametrize("field,value", [("sample_id", "wrong"), ("max_completion_tokens", 1),
    ("commit", "b"*40), ("success", False), ("sample_index", 999), ("repeat_id", "r009")])
def test_cell_raw_validation_rejects_drift(tmp_path, field, value):
    m, c, d = synthetic_valid_cell(tmp_path)
    rows = [json.loads(line) for line in (d / "raw.jsonl").read_text().splitlines()]
    rows[-1][field] = value
    (d / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises((ValueError, IndexError)):
        validate_cell(c, d, m)


from experiments.m4_treatments import definition
