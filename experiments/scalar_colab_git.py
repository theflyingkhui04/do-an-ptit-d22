"""Git-only Colab handoff and evidence transport; no model or GPU imports.

RUN_REPO is detached at a caller-supplied immutable SHA. Only EVIDENCE_REPO
may create/push an evidence commit. No merge, rebase, force push or retry.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from experiments.scalar_pinned_sources import manifest as pinned_manifest

REPO_URL = "https://github.com/theflyingkhui04/do-an-ptit-d22.git"
BRANCH = "feat/m4-02-04-scheduler-telemetry-v02"
RUN_PREFIX = "experiments/runs/m4_02_04/"
NOTEBOOK = "experiments/colab_m4_02_04_scalar_observer_git.ipynb"


def git(repo, *args, env=None):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + Path(repo).resolve().as_posix(),
         "-c", "core.quotepath=false", *args], cwd=repo, env=env, encoding="utf-8")


def checked_sha(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
        raise ValueError("Set CODE_SHA to the exact pushed 40-character SHA; no branch/HEAD fallback")
    return value


def checked_run_path(relative):
    if not re.fullmatch(re.escape(RUN_PREFIX) + r"[A-Za-z0-9][A-Za-z0-9_-]*", relative or ""):
        raise ValueError("Unsupported runtime evidence path")
    return relative


def source_gate(repo, code_sha, runtime_relative=None):
    repo, code_sha = Path(repo).resolve(), checked_sha(code_sha)
    if git(repo, "rev-parse", "HEAD").strip() != code_sha:
        raise ValueError("SOURCE GATE FAILED: HEAD != CODE_SHA")
    attached = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if attached.returncode != 1:
        raise ValueError("SOURCE GATE FAILED: RUN_REPO must be detached")
    specs = ["."]
    if runtime_relative is not None:
        specs.append(":(top,exclude)" + checked_run_path(runtime_relative) + "/")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", *specs)
    if dirty.strip():
        raise ValueError("SOURCE GATE FAILED: dirty source\n" + dirty)
    files = {}
    staged = git(repo, "ls-files", "--stage", "-z", "--", "src", "experiments", "tests", "scripts")
    for row in staged.split("\0"):
        if not row:
            continue
        meta, name = row.split("\t", 1)
        mode, oid, stage = meta.split()
        if name.startswith(("experiments/runs/", "experiments/results/")):
            continue
        path = repo / name
        if mode not in ("100644", "100755") or stage != "0" or path.is_symlink():
            raise ValueError("Unsupported source object: " + name)
        files[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), git_blob=oid)
    return dict(git_sha=code_sha, detached=True, source_clean=True,
        runtime_exclusion=runtime_relative, files=files,
        pinned_vllm_sources={row["path"]: row for row in pinned_manifest()["files"]})


@contextmanager
def github_auth(token):
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Missing Colab Secret: GITHUB_TOKEN")
    with tempfile.TemporaryDirectory(prefix="m40204-git-auth-") as directory:
        askpass = Path(directory) / "askpass.py"
        askpass.write_text("#!/usr/bin/env python3\nimport os,sys\n"
            "print('x-access-token' if 'username' in sys.argv[1].lower() "
            "else os.environ['M40204_GITHUB_TOKEN'])\n", encoding="utf-8")
        askpass.chmod(0o700)
        env = dict(os.environ, M40204_GITHUB_TOKEN=token, GIT_ASKPASS=str(askpass),
                   GIT_TERMINAL_PROMPT="0", GIT_CONFIG_COUNT="1",
                   GIT_CONFIG_KEY_0="credential.helper", GIT_CONFIG_VALUE_0="",
                   GIT_TRACE="0", GIT_TRACE_CURL="0", GIT_CURL_VERBOSE="0")
        try:
            yield env
        finally:
            env.pop("M40204_GITHUB_TOKEN", None)


def branch_gate(repo, *, env=None, expected_parent=None):
    if git(repo, "branch", "--show-current").strip() != BRANCH:
        raise ValueError("Evidence checkout is not on the writable M4-02-04 branch")
    git(repo, "fetch", "origin", f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", env=env)
    local = git(repo, "rev-parse", "HEAD").strip()
    remote = git(repo, "rev-parse", "refs/remotes/origin/" + BRANCH).strip()
    expected = local if expected_parent is None else checked_sha(expected_parent)
    if remote != expected:
        raise ValueError(f"EVIDENCE BRANCH DIVERGED: local={local}, remote={remote}; no automatic merge/rebase/push")
    return local


def push_evidence(run_repo, evidence_repo, code_sha, relative, *, env=None):
    run_repo, evidence_repo = Path(run_repo).resolve(), Path(evidence_repo).resolve()
    relative = checked_run_path(relative)
    if run_repo == evidence_repo or run_repo in evidence_repo.parents or evidence_repo in run_repo.parents:
        raise ValueError("RUN_REPO and EVIDENCE_REPO must be separate checkouts")
    source_gate(run_repo, code_sha, relative)
    if git(evidence_repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Evidence checkout must be clean before collection")
    parent = branch_gate(evidence_repo, env=env)
    source, target = run_repo / relative, evidence_repo / relative
    if not source.is_dir() or target.exists():
        raise ValueError("Missing run or existing evidence destination; no overwrite")
    source_files = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in evidence")
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".jsonl", ".log", ".txt", ".md", ".csv"):
            raise ValueError("Unsupported generated/binary evidence file: " + path.name)
        content = path.read_bytes()
        if re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+)", content):
            raise ValueError("Credential-like content in evidence; automatic staging refused")
        source_files[path.relative_to(source).as_posix()] = hashlib.sha256(content).hexdigest()
    if not source_files:
        raise ValueError("Empty run evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    for name, digest in source_files.items():
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Evidence copy changed bytes")
    git(evidence_repo, "add", "--", relative)
    staged = git(evidence_repo, "diff", "--cached", "--name-only", "-z").split("\0")
    staged = [name for name in staged if name]
    if not staged or any(not name.startswith(relative + "/") for name in staged):
        raise ValueError("Evidence staging escaped the explicit run directory")
    git(evidence_repo, "diff", "--cached", "--check")
    print(git(evidence_repo, "diff", "--cached", "--name-only"))
    print(git(evidence_repo, "diff", "--cached", "--stat"))
    from experiments.scalar_smoke_evidence import validate
    try:
        report = validate(source / "smoke")
    except Exception:
        report = {}
    verified = report.get("offline_validation") == "PASS" and report.get("verdict") == "LIVE OBSERVER VERIFIED"
    if verified:
        recorded = json.loads((source / "smoke/manifest.json").read_text())
        verified = recorded["git_sha"] == code_sha
    # Run validation before calling this function. Failed attempts are retained
    # with an explicit failure message; they never become a verified result.
    message = ("evidence(m4-02-04): record L4 scalar observer smoke" if verified else
               "evidence(m4-02-04): preserve failed L4 scalar observer smoke")
    git(evidence_repo, "config", "user.name", "M4 Colab Evidence")
    git(evidence_repo, "config", "user.email", "m4-colab-evidence@users.noreply.github.com")
    git(evidence_repo, "commit", "-m", message)
    commit = git(evidence_repo, "rev-parse", "HEAD").strip()
    branch_gate(evidence_repo, env=env, expected_parent=parent)
    git(evidence_repo, "push", "origin", f"HEAD:refs/heads/{BRANCH}", env=env)
    git(evidence_repo, "fetch", "origin", f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", env=env)
    if git(evidence_repo, "rev-parse", "refs/remotes/origin/" + BRANCH).strip() != commit:
        raise ValueError("Remote changed after push; preserve checkout for manual reconciliation")
    source_gate(run_repo, code_sha, relative)
    return dict(evidence_commit_sha=commit, evidence_push="PASS", verified=verified,
                run_code_sha=code_sha, staged_paths=staged)


def render_handoff(template, output, code_sha):
    checked_sha(code_sha)
    notebook = json.loads(Path(template).read_text(encoding="utf-8"))
    changed = 0
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if 'CODE_SHA = "SET_EXACT_PUSHED_SHA"' in text:
            text = text.replace('CODE_SHA = "SET_EXACT_PUSHED_SHA"', f'CODE_SHA = "{code_sha}"')
            changed += 1
        cell["source"] = text.splitlines(keepends=True)
    if changed != 1:
        raise ValueError("Template must contain exactly one explicit CODE_SHA placeholder")
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(notebook, stream, indent=2)
        stream.write("\n")
