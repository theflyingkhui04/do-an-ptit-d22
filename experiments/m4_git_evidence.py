"""User-run append-only Git evidence publishing with temporary askpass.

This module does nothing on import. SOURCE_COMMIT is never updated when the
evidence branch advances. No persistent credential URLs or credential files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from experiments.m4_protocol import git, sha256

BRANCH = "feat/m4-02-06-lossless-sweep-freeze"
REPO_URL = "https://github.com/theflyingkhui04/do-an-ptit-d22.git"


@contextmanager
def git_auth():
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("GITHUB_TOKEN must come from Colab Secrets; never place it in a remote URL")
    with tempfile.TemporaryDirectory(prefix="m4-askpass-") as directory:
        helper = Path(directory) / "askpass.py"
        helper.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "Username" in sys.argv[1] else os.environ["GITHUB_TOKEN"])\n')
        helper.chmod(0o700)
        yield dict(os.environ, GIT_ASKPASS=str(helper), GIT_TERMINAL_PROMPT="0")


def artifact_files(directory: Path):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Evidence must be a real directory")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in evidence")
        if not path.is_file():
            continue
        if (path.suffix.lower() not in (".json", ".jsonl", ".log", ".txt", ".csv", ".md")
                or path.stat().st_size > 40 * 1024**2):
            raise ValueError(f"Unexpected/oversized evidence file: {path.name}")
        data = path.read_bytes()
        if re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})", data):
            raise ValueError("Possible credential detected; evidence push refused")
        if any(os.environ.get(key) and os.environ[key].encode() in data for key in ("GITHUB_TOKEN", "HF_TOKEN")):
            raise ValueError("Secret value in artifact; evidence push refused")
        yield path


def copy_immutable(source: Path, destination: Path):
    files = list(artifact_files(source))  # Validate before copying any file.
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists() and sha256(target.read_bytes()) != sha256(path.read_bytes()):
            raise ValueError(f"Committed/previous evidence is immutable: {target}")
    for path in files:
        target = destination / path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def publish(directory: Path, evidence_repo: Path, source_commit: str) -> str:
    if not re.fullmatch(r"m4-02-06-[A-Za-z0-9_-]+", directory.name):
        raise ValueError("Invalid campaign/diagnostic ID")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Full SOURCE_COMMIT required")
    if git("branch", "--show-current", root=evidence_repo) != BRANCH:
        raise ValueError("Evidence checkout is on the wrong branch")
    if git("status", "--porcelain", root=evidence_repo):
        raise ValueError("Evidence checkout must be clean before synchronization")
    if git("remote", "get-url", "origin", root=evidence_repo) != REPO_URL:
        raise ValueError("Unexpected evidence remote; credential URLs are forbidden")
    def command(*args, env=None):
        subprocess.run(["git", *args], cwd=evidence_repo, env=env, check=True)
    with git_auth() as env:
        command("fetch", "origin", BRANCH, env=env)
        command("merge", "--ff-only", f"origin/{BRANCH}", env=env)
        command("merge-base", "--is-ancestor", source_commit, "HEAD")
        relative = Path("experiments/runs/m4_02_06") / directory.name
        copy_immutable(directory, evidence_repo / relative)
        command("add", "--", relative.as_posix())
        if git("diff", "--cached", "--name-only", root=evidence_repo):
            command("commit", "-m", f"evidence(m4-02-06): {directory.name}; source {source_commit[:12]}")
        command("push", "origin", f"HEAD:{BRANCH}", env=env)
    return git("rev-parse", "HEAD", root=evidence_repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--evidence-repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print("EVIDENCE_COMMIT=" + publish(args.directory, args.evidence_repo, args.source_commit))


if __name__ == "__main__":
    main()
