"""CPU regression of the three PR #43 CLI security boundaries."""
import json
import os
from pathlib import Path
import sys

import pytest

from experiments import m4_security as security
from experiments import m4_a100_profile as profile
from experiments import run_m4_colab_campaign as campaign
from experiments import m4_protocol as protocol

SHA = "a" * 40


@pytest.fixture
def work(tmp_path, monkeypatch):
    root = tmp_path / "content" / "m4-evidence-work"
    root.mkdir(parents=True)
    monkeypatch.setenv("M4_ARTIFACT_ROOT", str(root))
    return root


def directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows junctions need no symlink privilege and must be rejected too.
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def test_normal_colab_layout_and_relative_cwd_paths(work, monkeypatch):
    run = work / "m4-02-06-a100-profile-stamp-187cfff5"
    assert security.artifact_path(run, kind="new_directory") == run.resolve()
    run.mkdir()
    assert security.artifact_path(run, kind="directory") == run.resolve()
    path = run / "profile_sm80.json"
    path.write_text("{}")
    assert security.profile_path(path) == path.resolve()
    monkeypatch.chdir(work)
    assert security.profile_path(path.relative_to(work)) == path.resolve()


def test_default_handoff_work_and_preserved_evidence_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("M4_ARTIFACT_ROOT", raising=False)
    work = tmp_path / "content/m4-evidence-work"
    evidence = tmp_path / "content/do-an-ptit-d22-evidence/experiments/runs/m4_02_06"
    work.mkdir(parents=True)
    evidence.mkdir(parents=True)
    monkeypatch.setattr(security, "WORK_ROOT", work)
    monkeypatch.setattr(security, "EVIDENCE_ROOT", evidence)
    assert security.artifact_path(work / "setup/after-install", kind="new_directory") == work / "setup/after-install"
    path = evidence / "profile_sm80.json"
    path.write_text("{}")
    assert security.profile_path(path) == path
    with pytest.raises(ValueError, match="outside"):
        security.artifact_path(evidence / "new-output", kind="new_directory")


@pytest.mark.parametrize("case", ["parent", "absolute", "prefix", "normalized_parent"])
def test_out_escape_rejected(work, case, monkeypatch):
    monkeypatch.chdir(work)
    paths = {"parent": Path("../escape"), "absolute": work.parent / "escape",
             "prefix": work.with_name(work.name + "-evil") / "escape",
             "normalized_parent": work / "child/../escape"}
    with pytest.raises(ValueError):
        security.artifact_path(paths[case], kind="new_directory")


def test_output_must_be_new_and_worker_directory_must_exist(work):
    with pytest.raises(ValueError, match="below its approved root"):
        security.artifact_path(work, kind="new_directory")
    run = work / "existing"
    run.mkdir()
    with pytest.raises(ValueError, match="already exist"):
        security.artifact_path(run, kind="new_directory")
    with pytest.raises(FileNotFoundError):
        security.artifact_path(work / "missing", kind="directory")
    file = work / "file"
    file.write_text("{}")
    with pytest.raises(ValueError):
        security.artifact_path(file, kind="directory")


def test_output_symlink_ancestor_escape_and_approved_root_link_rejected(work, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = work / "redirect"
    directory_link(link, outside)
    with pytest.raises(ValueError, match="Symlinks/reparse"):
        security.artifact_path(link / "new-run", kind="new_directory")
    monkeypatch.setenv("M4_ARTIFACT_ROOT", str(link))
    with pytest.raises(ValueError, match="Symlinks/reparse"):
        security.artifact_path(link / "new-run", kind="new_directory")
    assert not (outside / "new-run").exists()


def test_configured_root_cannot_authorize_output_inside_source(work, monkeypatch):
    source = work / "run-checkout"
    source.mkdir()
    monkeypatch.setattr(security, "ROOT", source)
    with pytest.raises(ValueError, match="outside the frozen source"):
        security.artifact_path(source / "output", kind="new_directory")


@pytest.mark.parametrize("case", ["missing", "directory", "traversal", "outside", "wrong_name"])
def test_profile_gate_rejects_bad_paths_before_content_sink(work, tmp_path, case, monkeypatch):
    directory = work / "directory/profile_sm80.json"
    directory.mkdir(parents=True)
    wrong = work / "other.json"
    wrong.write_text("{}")
    outside = tmp_path / "profile_sm80.json"
    outside.write_text("{}")
    paths = {"missing": work / "missing/profile_sm80.json", "directory": directory,
             "traversal": work / "../profile_sm80.json", "outside": outside, "wrong_name": wrong}
    monkeypatch.setattr(campaign, "validate_profile_file", lambda *_: pytest.fail("Unsafe path reached content validator"))
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.profile_gate(paths[case], SHA)


def test_profile_symlink_escape_rejected_at_both_sinks(work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "profile_sm80.json").write_text("{}")
    link = work / "redirect"
    directory_link(link, outside)
    for validate in (profile.validate_profile_file, campaign.profile_gate):
        with pytest.raises(ValueError, match="Symlinks/reparse"):
            validate(link / "profile_sm80.json", SHA)


def test_profile_regular_file_symlink_rejected(work, tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("{}")
    link = work / "profile_sm80.json"
    try:
        link.symlink_to(target)
    except OSError:
        if os.name != "nt":
            raise
        pytest.skip("File symlinks require Windows symlink privilege; directory reparse escapes tested separately")
    with pytest.raises(ValueError, match="Symlinks/reparse"):
        security.profile_path(link)


def test_sibling_raw_symlink_escape_rejected(work, tmp_path, monkeypatch):
    # A valid profile path alone must not authorize its raw/env sibling links.
    run = work / "run"
    run.mkdir()
    (run / "profile_sm80.json").write_text("{}")
    monkeypatch.setattr(profile, "validate_profile", lambda *_: None)
    outside = tmp_path / "outside"
    outside.mkdir()
    directory_link(run / "raw.jsonl", outside)
    with pytest.raises(ValueError, match="Symlinks/reparse"):
        profile.validate_profile_file(run / "profile_sm80.json", SHA)


@pytest.mark.parametrize("value", ["--help", "-" + "a" * 39, "A" * 40, "a" * 39,
    "a" * 41, "abc;rm -rf /", "a" * 39 + "\n", " " + "a" * 40, "a" * 39 + "\0"])
def test_source_commit_rejects_injection_before_git(value, monkeypatch):
    monkeypatch.setattr(protocol, "git", lambda *_args, **_kwargs: pytest.fail("Malformed SHA reached Git"))
    with pytest.raises(ValueError, match="SOURCE_GATE FAIL"):
        security.validated_source_commit(value)


@pytest.mark.parametrize("head,dirty,accepted", [(SHA, "", True), ("b" * 40, "", False),
                                               (SHA, " M tracked.py", False)])
def test_source_commit_preserves_exact_head_and_clean_tree_gate(monkeypatch, head, dirty, accepted):
    def git(*args, **kwargs):
        return head if args[0] == "rev-parse" else dirty
    monkeypatch.setattr(protocol, "git", git)
    if accepted:
        assert security.validated_source_commit(SHA) == SHA
    else:
        with pytest.raises(ValueError, match="SOURCE_GATE FAIL"):
            security.validated_source_commit(SHA)


def test_worker_subprocess_uses_only_returned_validated_sha(work, monkeypatch):
    from experiments import m4_runtime
    trusted = "b" * 40
    calls = []
    monkeypatch.setattr(sys, "argv", ["profile", "measure", "--source-commit", SHA, "--out", str(work / "new-run")])
    monkeypatch.setattr(profile, "validated_source_commit", lambda value: trusted)
    monkeypatch.setattr(profile, "inspect_environment", lambda: {})
    monkeypatch.setattr(profile, "hardware_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(profile, "KS", (0,))
    monkeypatch.setattr(profile, "BATCHES", (1,))
    monkeypatch.setattr(profile, "build_profile", lambda env, *_: {"source_commit": env["source_commit"]})
    monkeypatch.setattr(m4_runtime, "gpu_snapshot", lambda: {"processes": ""})
    monkeypatch.setattr(m4_runtime, "require_gpu_release", lambda *_: None)
    def worker(command, **kwargs):
        calls.append(command)
        assert command[command.index("--source-commit") + 1] == trusted
        assert SHA not in command and not kwargs.get("shell", False)
        part = Path(command[command.index("--out") + 1]) / "k0-b1-c1024"
        part.mkdir()
        (part / "raw.jsonl").write_text('{"synthetic":true}\n')
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(profile.subprocess, "run", worker)
    profile.main()
    assert len(calls) == 1
    assert json.loads((work / "new-run/env.json").read_text())["source_commit"] == trusted


def test_invalid_out_fails_before_environment_probe(work, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["profile", "environment", "--source-commit", SHA, "--out", str(work / "../escape")])
    monkeypatch.setattr(profile, "validated_source_commit", lambda _: SHA)
    monkeypatch.setattr(profile, "inspect_environment", lambda: pytest.fail("Unsafe output reached hardware probe"))
    with pytest.raises(ValueError, match="traversal"):
        profile.main()


def test_validate_cli_out_rejects_outside_file(work, tmp_path, monkeypatch):
    outside = tmp_path / "profile_sm80.json"
    outside.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["profile", "validate", "--source-commit", SHA, "--out", str(outside)])
    with pytest.raises(ValueError, match="outside"):
        profile.main()


def test_reproducible_profile_passes_both_cli_gates(diagnostic_fixture, monkeypatch):
    path = diagnostic_fixture / "profile_inputs/profile_sm80.json"
    assert campaign.profile_gate(path, SHA) == protocol.sha256(path.read_bytes())
    # Offline validation retains its original identity check against the profile,
    # rather than requiring the validating checkout HEAD to equal the old run.
    monkeypatch.setattr(sys, "argv", ["profile", "validate", "--source-commit", SHA, "--out", str(path)])
    profile.main()


from test_m4_readiness import diagnostic_fixture
