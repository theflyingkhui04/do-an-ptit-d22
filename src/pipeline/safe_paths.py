"""Command-line paths of the workload tools: inside the repository or the system temp directory only.

Relative paths resolve against the repository root, so a script behaves the same from any working
directory. Symlinks and ``..`` are resolved first, then the result must sit below an approved root;
anything else (another drive, a system folder) is refused before it is read or written.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def approved_roots() -> tuple[str, ...]:
    return (os.path.realpath(REPO_ROOT), os.path.realpath(tempfile.gettempdir()))


def contained_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a command-line path and refuse it outside the repository and the temp directory."""
    candidate = os.path.realpath(os.path.join(REPO_ROOT, os.fspath(value)))
    for root in approved_roots():
        try:
            if os.path.commonpath([root, candidate]) == root:
                return Path(candidate)
        except ValueError:  # different drives on Windows
            continue
    raise ValueError(f"Path outside the repository and the temp directory: {value}")
