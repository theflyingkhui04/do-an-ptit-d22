"""M4 CLI trust boundaries; no filesystem or process changes on import.

The Colab handoff writes under /content/m4-evidence-work. Operators using a
different WORK directory must provision it first and set M4_ARTIFACT_ROOT in
the launcher environment (never inferred from an untrusted path argument).
Preserved profiles may also be read from the handoff's evidence checkout.
These checks assume no concurrent hostile writer in those operator-owned roots.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from experiments.m4_protocol import ROOT, source_gate

WORK_ROOT = Path("/content/m4-evidence-work")
EVIDENCE_ROOT = Path("/content/do-an-ptit-d22-evidence/experiments/runs/m4_02_06")


def _work_root() -> Path:
    configured = os.environ.get("M4_ARTIFACT_ROOT")
    if configured is not None and not Path(configured).is_absolute():
        raise ValueError("M4_ARTIFACT_ROOT must be an absolute operator-owned directory")
    return Path(configured).absolute() if configured is not None else WORK_ROOT.absolute()


def _reject_links(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Symlinks/reparse points are forbidden in M4 artifact paths")


def artifact_path(value: Path, *, kind: str) -> Path:
    """Return a contained canonical regular file, existing dir or new dir path.

    Relative paths retain normal cwd semantics. '..' is rejected even if it
    would normalize back inside the root. New output directories cannot replace
    an existing object; worker directories must already exist.
    """
    if kind not in ("file", "directory", "new_directory"):
        raise ValueError("Unknown M4 artifact path kind")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden in M4 artifact paths")
    path = path.absolute()
    if kind != "file":
        try:
            path.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise ValueError("M4 output must be outside the frozen source checkout")
    roots = (_work_root(),) if kind != "file" else (_work_root(), EVIDENCE_ROOT.absolute())
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            raise ValueError("An artifact must be below its approved root")
        _reject_links(root)
        _reject_links(path)
        approved = root.resolve(strict=True)
        if not approved.is_dir():
            raise ValueError("M4 artifact root must be a directory")
        resolved = path.resolve(strict=kind != "new_directory")
        resolved.relative_to(approved)
        if kind == "new_directory":
            if resolved.exists():
                raise ValueError("M4 output directory must not already exist")
        elif kind == "file" and not resolved.is_file():
            raise ValueError("M4 profile inputs must be regular files")
        elif kind == "directory" and not resolved.is_dir():
            raise ValueError("M4 worker output must be an existing directory")
        return resolved
    raise ValueError("M4 artifact path is outside the approved roots")


def profile_path(value: Path) -> Path:
    path = artifact_path(value, kind="file")
    if path.name != "profile_sm80.json":
        raise ValueError("M4 requires profile_sm80.json with sibling env.json/raw.jsonl")
    return path


def validated_source_commit(value: str) -> str:
    """Return only a strict SHA after exact HEAD and clean-checkout validation."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("SOURCE_GATE FAIL: full lowercase hexadecimal SHA required")
    trusted_sha = bytes.fromhex(value).hex()
    source_gate(trusted_sha)
    return trusted_sha
