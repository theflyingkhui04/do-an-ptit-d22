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
