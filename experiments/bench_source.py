"""Package committed benchmark bytes independently of checkout line endings."""
from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path


def snapshot(root: Path, commit: str = "HEAD") -> tuple[tempfile.TemporaryDirectory, str]:
    if commit != "HEAD" and not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source snapshot requires HEAD or a full immutable SHA")
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]
    sha = subprocess.check_output([*git, "rev-parse", commit], cwd=root, text=True).strip()
    tree = subprocess.check_output([
        *git, "ls-tree", "-r", "-z", sha, "--", "src", "experiments",
    ], cwd=root)
    selected = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        path = name.decode("utf-8")
        if path.startswith(("experiments/runs/", "experiments/results/", "experiments/notebooks/")):
            continue
        if "__pycache__" in path or path.endswith(".ipynb"):
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError(f"Unsupported source entry: {path}")
        selected.append((path, object_id))
    # Unlike git archive, cat-file never applies core.autocrlf or export filters.
    blobs = subprocess.run([*git, "cat-file", "--batch"], cwd=root,
                           input=b"".join(oid + b"\n" for _, oid in selected),
                           stdout=subprocess.PIPE, check=True).stdout
    directory = tempfile.TemporaryDirectory(prefix="ptit-committed-")
    stream = io.BytesIO(blobs)
    for path, oid in selected:
        actual_oid, kind, size = stream.readline().split()
        if actual_oid != oid or kind != b"blob":
            raise ValueError("Git blob packaging mismatch")
        content = stream.read(int(size))
        if stream.read(1) != b"\n" or len(content) != int(size):
            raise ValueError("Truncated source blob")
        target = Path(directory.name) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return directory, sha
