"""Local-only source bundle and immutable plan preparation. Never imports Modal."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from experiments.eagle_pulse_campaign import (
    ROOT, BRANCH, atomic_json, canonical, digest, git, initialize, read_json,
    require, sha_file, source_gate, validate,
)
from experiments.eagle_pulse_campaign_fake import make_plan


def freeze(campaign_dir, max_allocations, synthetic=False):
    root = Path(campaign_dir).resolve()
    sha = git(ROOT, "rev-parse", "HEAD")
    source_gate(ROOT, sha)
    require(not root.exists(), "freeze_destination_exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulse-source-", dir=root.parent) as temporary:
        staging = Path(temporary)
        require(staging.resolve().is_relative_to(root.parent), "temporary_source_directory")
        bundle = staging / "reviewed-source.bundle"
        git(ROOT, "bundle", "create", str(bundle), "refs/heads/" + BRANCH)
        bundle_output = git(ROOT, "bundle", "verify", str(bundle))
        checkout = staging / "empty-reconstruction"
        require(not checkout.exists(), "reconstruction_must_start_empty")
        subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.longpaths=true", "clone", "--quiet",
                        "--branch", BRANCH, str(bundle), str(checkout)], check=True)
        require(git(checkout, "rev-parse", "HEAD") == sha, "bundle_head")
        require(not git(checkout, "status", "--short"), "bundle_dirty_checkout")
        fsck_output = git(checkout, "fsck", "--full")
        require(not (checkout / ".git/objects/info/alternates").exists(), "bundle_alternates")
        names = git(ROOT, "ls-files", "src", "experiments", "tests", "pyproject.toml").splitlines()
        files = {}
        for name in names:
            if Path(name).suffix not in (".py", ".json", ".toml"):
                continue
            committed = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "show", sha + ":" + name], cwd=ROOT)
            loaded = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            require(committed == loaded == (checkout / name).read_bytes(), "source_manifest:" + name)
            files[name] = hashlib.sha256(committed).hexdigest()
        package = dict(schema="pulse-source-package-v1", source_git_sha=sha, branch=BRANCH,
                       bundle_sha256=sha_file(bundle), bundle_bytes=bundle.stat().st_size, files=files)
        verdict = dict(source_git_sha=sha, verified=True, bundle_verify=True, empty_reconstruction=True,
                       exact_head=True, clean_checkout=True, fsck_full=True, no_alternates=True,
                       runtime_manifest_files=len(files), bundle_sha256=package["bundle_sha256"], bundle_bytes=package["bundle_bytes"])
        verdict.update(reconstruction_started_empty=True, reconstruction_head=git(checkout, "rev-parse", "HEAD"),
                       reconstruction_status=git(checkout, "status", "--short"),
                       bundle_verify_output=bundle_output, fsck_output=fsck_output)
        plan = make_plan(sha, max_allocations, synthetic)
        plan.update(source_bundle_sha256=package["bundle_sha256"], source_package_sha256=digest(package))
        initialize(root, plan, ("pulse-dual-synthetic-v1-" if synthetic else "pulse-dual-live-v1-") + digest(plan))
        source = root / "source"
        source.mkdir()
        shutil.copyfile(bundle, source / "reviewed-source.bundle")
        (source / "bundle.sha256").write_text(package["bundle_sha256"] + "\n", encoding="utf-8")
        atomic_json(source / "package-manifest.json", package)
        atomic_json(source / "source-verdict.json", verdict)
    validate(root)
    return dict(verdict, campaign_dir=str(root), plan_sha256=digest(plan))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--max-allocations", type=int, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.campaign_dir, args.max_allocations, args.synthetic), indent=2))


if __name__ == "__main__":
    main()
