"""Local Git-only preflight. No GitHub, Modal, model import or GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from experiments import scalar_colab_git as flow
from experiments import scalar_pinned_sources as pinned


def command(repo, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + Path(repo).resolve().as_posix(),
        *args], cwd=repo, encoding="utf-8", stderr=subprocess.STDOUT).strip()


@pytest.fixture
def checkouts(tmp_path):
    remote, seed = tmp_path / "remote.git", tmp_path / "seed"
    remote.mkdir(); seed.mkdir()
    command(remote, "init", "--bare")
    command(seed, "init", "-b", flow.BRANCH)
    command(seed, "config", "user.name", "Test")
    command(seed, "config", "user.email", "test@example.invalid")
    (seed / "src").mkdir()
    (seed / "src/code.py").write_text("value = 1\n")
    command(seed, "add", "--", "src/code.py")
    command(seed, "commit", "-m", "fixture source")
    command(seed, "remote", "add", "origin", str(remote))
    command(seed, "push", "origin", flow.BRANCH)
    run, evidence = tmp_path / "run", tmp_path / "evidence"
    command(tmp_path, "clone", "--branch", flow.BRANCH, str(remote), str(run))
    command(tmp_path, "clone", "--branch", flow.BRANCH, str(remote), str(evidence))
    sha = command(run, "rev-parse", "HEAD")
    command(run, "checkout", "--detach", sha)
    return NS(remote=remote, seed=seed, run=run, evidence=evidence, sha=sha,
              relative=flow.RUN_PREFIX + "synthetic-local-test")


def failed_run(state):
    root = state.run / state.relative
    root.mkdir(parents=True)
    (root / "failure.json").write_text(json.dumps({"synthetic": True, "error": "retained fake failure"}))
    return root


def advance_remote(state):
    (state.seed / "later.txt").write_text("another writer\n")
    command(state.seed, "add", "--", "later.txt")
    command(state.seed, "commit", "-m", "remote advance")
    command(state.seed, "push", "origin", flow.BRANCH)


def test_source_gate_requires_detached_exact_clean_checkout(checkouts):
    s = checkouts
    assert flow.source_gate(s.run, s.sha)["source_clean"]
    with pytest.raises(ValueError, match="detached"):
        flow.source_gate(s.evidence, s.sha)
    with pytest.raises(ValueError, match="HEAD != CODE_SHA"):
        flow.source_gate(s.run, "0" * 40)
    failed_run(s)
    assert flow.source_gate(s.run, s.sha, s.relative)["source_clean"]
    (s.run / "unrelated.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="dirty source"):
        flow.source_gate(s.run, s.sha, s.relative)


def test_production_change_is_never_excluded_by_runtime_path(checkouts):
    s = checkouts
    failed_run(s)
    (s.run / "src/code.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="dirty source"):
        flow.source_gate(s.run, s.sha, s.relative)


def test_failure_evidence_push_changes_only_evidence_checkout(checkouts):
    s = checkouts
    failed_run(s)
    result = flow.push_evidence(s.run, s.evidence, s.sha, s.relative)
    assert result["evidence_push"] == "PASS" and result["verified"] is False
    assert command(s.run, "rev-parse", "HEAD") == s.sha
    assert command(s.evidence, "rev-parse", "HEAD") != s.sha
    assert "preserve failed" in command(s.evidence, "log", "-1", "--format=%s")
    assert all(path.startswith(s.relative + "/") for path in result["staged_paths"])


def test_validated_synthetic_evidence_push_binds_actual_code_sha(checkouts):
    from tests.test_scalar_colab_smoke import synthetic_smoke
    from experiments import scalar_colab_smoke as smoke
    s = checkouts
    root = s.run / s.relative
    root.mkdir(parents=True)
    child = synthetic_smoke(root / "smoke")
    for name in ("manifest.json", "source_manifest.json"):
        data = json.loads((child / name).read_text())
        data["git_sha"] = s.sha
        smoke.save(child / name, data)
    smoke.seal(child)
    result = flow.push_evidence(s.run, s.evidence, s.sha, s.relative)
    assert result["verified"] and result["run_code_sha"] == s.sha
    assert command(s.run, "rev-parse", "HEAD") == s.sha


def test_remote_advance_blocks_before_staging(checkouts):
    s = checkouts
    failed_run(s)
    advance_remote(s)
    with pytest.raises(ValueError, match="DIVERGED"):
        flow.push_evidence(s.run, s.evidence, s.sha, s.relative)
    assert command(s.evidence, "rev-parse", "HEAD") == s.sha
    assert not command(s.evidence, "diff", "--cached", "--name-only")


def test_remote_race_preserves_local_evidence_commit_without_push(checkouts, monkeypatch):
    s = checkouts
    failed_run(s)
    original = flow.branch_gate
    def gate(*args, **kwargs):
        if kwargs.get("expected_parent") is not None:
            advance_remote(s)
        return original(*args, **kwargs)
    monkeypatch.setattr(flow, "branch_gate", gate)
    with pytest.raises(ValueError, match="DIVERGED"):
        flow.push_evidence(s.run, s.evidence, s.sha, s.relative)
    assert command(s.evidence, "rev-parse", "HEAD") != s.sha
    assert command(s.evidence, "rev-parse", "HEAD") != command(s.seed, "rev-parse", "HEAD")
    assert command(s.run, "rev-parse", "HEAD") == s.sha


def test_foreign_staged_file_blocks_evidence_commit(checkouts):
    s = checkouts
    failed_run(s)
    (s.evidence / "unrelated.py").write_text("x = 1\n")
    command(s.evidence, "add", "--", "unrelated.py")
    with pytest.raises(ValueError, match="must be clean"):
        flow.push_evidence(s.run, s.evidence, s.sha, s.relative)


@pytest.mark.parametrize("relative", ["../escape", flow.RUN_PREFIX + "a/../../src", "src/code.py", flow.RUN_PREFIX])
def test_runtime_path_cannot_escape_exclusion(relative):
    with pytest.raises(ValueError):
        flow.checked_run_path(relative)


def test_generated_binary_and_credentials_are_not_staged(checkouts):
    s = checkouts
    root = failed_run(s)
    bad = root / "capture.zip"
    bad.write_bytes(b"not evidence")
    with pytest.raises(ValueError, match="binary"):
        flow.push_evidence(s.run, s.evidence, s.sha, s.relative)
    bad.unlink()
    (root / "secret.txt").write_text("ghp_" + "a" * 32)
    with pytest.raises(ValueError, match="Credential"):
        flow.push_evidence(s.run, s.evidence, s.sha, s.relative)


def test_auth_script_has_no_embedded_secret_and_is_removed():
    token = "synthetic-auth-value"
    with flow.github_auth(token) as env:
        path = Path(env["GIT_ASKPASS"])
        script = path.read_text()
        assert token not in script and env["M40204_GITHUB_TOKEN"] == token
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        compile(script, "askpass.py", "exec")
    assert not path.exists()


@pytest.mark.parametrize("mutation", [None, "version", "hash"])
def test_installed_source_fallback_is_hash_verified_without_vllm_import(tmp_path, monkeypatch, mutation):
    name = "vllm/v1/core/sched/utils.py"
    try:
        content = pinned.read_source(name)
    except FileNotFoundError as error:  # no vllm==0.30.0 and no local capture (as in test_scalar_cost_capture)
        pytest.skip(str(error))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(content if mutation != "hash" else b"changed source")
    monkeypatch.setattr(pinned, "CAPTURE", tmp_path / "absent")
    monkeypatch.setattr(pinned.importlib.metadata, "distribution", lambda _: NS(
        version="other" if mutation == "version" else "0.30.0", locate_file=lambda value: tmp_path / value))
    if mutation:
        with pytest.raises(ValueError, match="MISMATCH"):
            pinned.read_source(name)
    else:
        assert pinned.read_source(name) == content


@pytest.mark.parametrize("mutation", [None, "sha", "command", "log"])
def test_reused_pre_gpu_gate_requires_same_sha_interpreter_and_log_hashes(tmp_path, monkeypatch, mutation):
    from experiments import scalar_colab_smoke as smoke
    previous, target = tmp_path / "preflight", tmp_path / "run"
    previous.mkdir(); target.mkdir()
    for name in ("pre_gpu_tests.log", "colab_preflight_tests.log"):
        (previous / name).write_text("SYNTHETIC CPU TEST FIXTURE ONLY")
    report = dict(git_sha="a" * 40, argv=[sys.executable, "-m", "pytest", "-q", *smoke.TESTS, "-rA"],
        exit_code=0, passed=230, skipped=0, log_sha256=smoke.sha256(previous / "pre_gpu_tests.log"))
    small = dict(git_sha="a" * 40, exit_code=0, log_sha256=smoke.sha256(previous / "colab_preflight_tests.log"))
    if mutation == "sha": report["git_sha"] = "b" * 40
    if mutation == "command": report["argv"] = ["other-python"]
    if mutation == "log": (previous / "pre_gpu_tests.log").write_text("changed")
    smoke.save(previous / "pre_gpu_tests.json", report)
    smoke.save(previous / "colab_preflight_tests.json", small)
    monkeypatch.setattr(smoke, "source_gate", lambda: {"git_sha": "a" * 40})
    if mutation:
        with pytest.raises(RuntimeError): smoke.reuse_preflight(target, previous)
    else:
        smoke.reuse_preflight(target, previous)
        assert (target / "pre_gpu_tests.log").read_bytes() == (previous / "pre_gpu_tests.log").read_bytes()


def test_failed_pre_gpu_gate_cannot_launch_cell_or_model(tmp_path, monkeypatch):
    from experiments import scalar_colab_smoke as smoke
    monkeypatch.setenv("PTIT_SCALAR_CODE_SHA", "a" * 40)
    monkeypatch.setattr(smoke, "source_gate", lambda: {"git_sha": "a" * 40})
    monkeypatch.setattr(smoke, "environment_gate", lambda: {"synthetic": True})
    calls = []
    def subprocess_run(argv, **kwargs):
        calls.append(argv)
        assert argv[1:] == ["-m", "pip", "freeze"], "No model/cell may launch after failed preflight"
        return NS(stdout="synthetic\n")
    monkeypatch.setattr(smoke.subprocess, "run", subprocess_run)
    def fail(_): raise RuntimeError("synthetic pre-GPU failure")
    monkeypatch.setattr(smoke, "pre_gpu_gate", fail)
    root = tmp_path / "failed-gate"
    assert smoke.run(root) is False
    assert (root / "failure.json").exists()
    assert json.loads((root / "validation.json").read_text())["verdict"] == "LIVE OBSERVER NOT VERIFIED"
    assert len(calls) == 1
