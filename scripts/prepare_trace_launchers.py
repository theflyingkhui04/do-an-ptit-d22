"""Create diagnostic launchers from local canonical sources without changing them.

Run from the repository root:
    python scripts/prepare_trace_launchers.py
"""
from pathlib import Path
from difflib import unified_diff
import hashlib
import os

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

PLANS = [
    (
        SCRIPTS / "launch_lane3_adaptive.sh",
        SCRIPTS / "launch_lane3_adaptive_trace_diag.sh",
        [
            (
                "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                "src.server.adaptive_scheduler_trace_diag.TraceAdaptiveScheduler",
                1,
            ),
            (
                "--per-request-spec-decode-metrics summary",
                "--per-request-spec-decode-metrics detailed",
                1,
            ),
        ],
    ),
    (
        SCRIPTS / "switch_lane.sh",
        SCRIPTS / "switch_lane_trace_diag.sh",
        [
            (
                "launch_lane3_adaptive.sh",
                "launch_lane3_adaptive_trace_diag.sh",
                2,
            )
        ],
    ),
]

prepared = []
for source, destination, replacements in PLANS:
    if destination.exists():
        raise SystemExit(f"STOP: refusing to overwrite {destination}")
    original = source.read_text(encoding="utf-8")
    content = original
    for old, new, expected in replacements:
        count = content.count(old)
        if count != expected:
            raise SystemExit(
                f"STOP: {source}: {old!r} count={count}, expected={expected}"
            )
        content = content.replace(old, new)
    prepared.append((source, destination, original, content))

for source, destination, original, content in prepared:
    with destination.open("x", encoding="utf-8") as output:
        output.write(content)
    os.chmod(destination, source.stat().st_mode & 0o777)
    print(f"CREATED: {destination}")
    print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    print("\n".join(unified_diff(
        original.splitlines(), content.splitlines(),
        fromfile=str(source), tofile=str(destination), lineterm=""
    )))

print("CANONICAL_SCRIPTS_UNCHANGED")
