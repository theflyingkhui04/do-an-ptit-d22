"""[M2-03] The Protocol B diagnostics ran against these sources as they were at FROZEN_COMMIT (19/10).

M4-02 later changed them on purpose (async lineage, scalar budget, FlexSDS CLI). The diagnostic code keeps
fail-closed gates that read the live files, so it can only execute from a tree that still holds the frozen
sources. Tests use this module in two ways:

* source/logic checks read the frozen bytes (``frozen_bytes``) instead of the live file, so the approval
  rules stay tested on every branch;
* checks that need the live tree to be the M2-03 tree skip with ``LIVE_TREE_REASON`` when it is not.

Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "529e7f5e3deb569638473d0a63cd48d7e9e47c4b"  # diagnostic(m2-03): add position177 k0 control
FROZEN_PATHS = ("src/server/adaptive_scheduler.py", "src/server/budget_planner.py", "src/proposer/controller.py",
                "src/proposer/adapter.py", "scripts/launch_lane3_adaptive.sh")


@functools.cache
def frozen_bytes(path: str) -> bytes:
    """Bytes of ``path`` at FROZEN_COMMIT, read from Git (never from the working tree)."""
    if path not in FROZEN_PATHS:
        raise ValueError(f"not an M2-03 frozen source: {path}")
    return subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
                                    f"{FROZEN_COMMIT}:{path}"], cwd=ROOT)


@functools.cache
def live_drift() -> tuple[str, ...]:
    """Frozen sources whose live bytes differ from FROZEN_COMMIT (empty when the M2-03 tree is checked out)."""
    return tuple(path for path in FROZEN_PATHS
                 if not (ROOT / path).is_file() or (ROOT / path).read_bytes() != frozen_bytes(path))


def live_tree_reason() -> str:
    return (f"M2-03 diagnostic needs the frozen sources live; changed since {FROZEN_COMMIT[:7]}: "
            + ", ".join(live_drift()) + f" (run from {FROZEN_COMMIT[:7]})")
