"""Post-checkpoint local gate. Set PULSE_HANDOFF_DIR to the prepared live root."""
import os
from pathlib import Path
import pytest

import experiments.eagle_pulse_campaign as c


@pytest.fixture(scope="module")
def handoff():
    root = Path(os.environ["PULSE_HANDOFF_DIR"])
    plan = c.read_json(root / "campaign-plan.json")
    package = c.read_json(root / "source/package-manifest.json")
    verdict = c.read_json(root / "source/source-verdict.json")
    return root, plan, package, verdict


@pytest.mark.parametrize("identifier", [f"CAM-N{i:02d}" for i in range(1, 11)])
def test_local_source_handoff(handoff, identifier):
    root, plan, package, verdict = handoff
    bundle = root / "source/reviewed-source.bundle"
    sha = c.git(c.ROOT, "rev-parse", "HEAD")
    if identifier == "CAM-N01":
        assert sha == plan["source_git_sha"] == package["source_git_sha"] == verdict["source_git_sha"]
    elif identifier == "CAM-N02":
        refs = c.git(c.ROOT, "bundle", "list-heads", str(bundle))
        assert sha + " refs/heads/" + c.BRANCH in refs
    elif identifier == "CAM-N03":
        assert "complete history" in c.git(c.ROOT, "bundle", "verify", str(bundle))
    elif identifier == "CAM-N04":
        assert verdict["empty_reconstruction"] and verdict["reconstruction_started_empty"]
    elif identifier == "CAM-N05":
        assert verdict["reconstruction_head"] == sha
    elif identifier == "CAM-N06":
        assert verdict["clean_checkout"] and verdict["reconstruction_status"] == ""
    elif identifier == "CAM-N07":
        assert verdict["fsck_full"]
        assert "missing" not in verdict["fsck_output"].lower()
    elif identifier == "CAM-N08":
        assert verdict["no_alternates"]
    elif identifier == "CAM-N09":
        assert c.verify_source(root, plan) == package
        assert len(package["files"]) == verdict["runtime_manifest_files"]
        assert "experiments/eagle_pulse_campaign_worker.py" in package["files"]
    elif identifier == "CAM-N10":
        assert c.sha_file(bundle) == package["bundle_sha256"] == verdict["bundle_sha256"]
        assert bundle.stat().st_size == package["bundle_bytes"] == verdict["bundle_bytes"]


def test_freeze_normalizes_windows_checkout(tmp_path, monkeypatch):
    """Git may check out CRLF while committed canonical text uses LF."""
    import subprocess
    import shutil
    import experiments.eagle_pulse_campaign_freeze as f
    original_root = c.ROOT
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "experiments").mkdir()
    (repo / "src").mkdir()
    shutil.copyfile(original_root / "experiments/eagle_pulse_campaign_contract.json",
                    repo / "experiments/eagle_pulse_campaign_contract.json")
    (repo / "src/example.py").write_bytes(b"VALUE = 1\r\n")
    (repo / "src/legacy.py").write_bytes(b"LEGACY = 2\r\n")
    (repo / "pyproject.toml").write_bytes(b"[project]\r\nname = 'cpu-source-fixture'\r\n")
    (repo / ".gitattributes").write_text("* text eol=crlf\nsrc/legacy.py -text\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.longpaths")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    def git(*args):
        return subprocess.check_output(["git", "-c", "safe.directory=" + repo.as_posix(),
            "-c", "user.name=CPU Fixture", "-c", "user.email=cpu@example.invalid",
            "-c", "commit.gpgsign=false", "-C", str(repo), *args])
    git("init", "--quiet", "--initial-branch=" + c.BRANCH)
    git("add", "--", ".gitattributes", "src/example.py", "src/legacy.py", "pyproject.toml",
        "experiments/eagle_pulse_campaign_contract.json")
    git("commit", "--quiet", "-m", "CPU source fixture")
    assert b"\r\n" not in git("show", "HEAD:src/example.py")
    assert b"\r\n" in git("show", "HEAD:src/legacy.py")
    monkeypatch.setattr(c, "ROOT", repo)
    monkeypatch.setattr(f, "ROOT", repo)
    monkeypatch.setattr(f, "validate", lambda root: c.validate(root, checkout=repo))
    result = f.freeze(tmp_path / "campaign", 1, synthetic=True)
    assert result["verified"] and result["clean_checkout"]
    assert result["source_normalization"] == "git_lf_text_v1"
    assert result["empty_reconstruction"] and result["no_alternates"]
    plan, state = c.validate(tmp_path / "campaign", checkout=repo)
    assert state["status"] == "READY" and state["allocation_count"] == 0
