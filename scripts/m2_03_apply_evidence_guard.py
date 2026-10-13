#!/usr/bin/env python3
"""Apply a fail-closed, minimal evidence-overwrite guard to M2-03 verifier.

Dry-run by default. --apply creates a byte-for-byte backup before editing.
This tool edits only tests/verify_lossless.py; it never reads/writes model results.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from pathlib import Path


ANCHOR = (
    "def verify_stage(run_id: str) -> int:\n"
    "    run_dir = RUN_ROOT / run_id\n"
)
GUARD = (
    "    # Reject reusing a run ID before opening a session, contacting a server,\n"
    "    # or entering the try/finally block that writes canonical summaries.\n"
    "    # Preserve both failed and successful Adaptive evidence without exception.\n"
    "    existing_evidence = [\n"
    "        path.name\n"
    "        for path in (run_dir / \"adaptive.jsonl\", run_dir / \"summary.json\")\n"
    "        if path.exists()\n"
    "    ]\n"
    "    if existing_evidence:\n"
    "        print(\n"
    "            f\"REFUSING_REUSED_RUN_ID: {run_id}; existing evidence: \"\n"
    "            + \", \".join(existing_evidence),\n"
    "            file=sys.stderr,\n"
    "        )\n"
    "        return 1\n"
    "\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Save a backup and apply exact guard")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (for tests; defaults to this script's parent repo)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    target = root / "tests/verify_lossless.py"
    if not target.is_file():
        parser.error(f"Missing target {target}")
    raw = target.read_bytes()
    # Preserve LF source as-is; fail closed rather than silently normalizing files.
    try:
        current = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        parser.error(f"Target must be UTF-8: {exc}")
    if current.count(ANCHOR) != 1:
        parser.error("Expected exactly one verify_stage() anchor; refused to edit")
    if "REFUSING_REUSED_RUN_ID" in current:
        parser.error("Evidence guard appears to be already applied; no change")
    candidate = current.replace(ANCHOR, ANCHOR + GUARD, 1)
    if candidate.count("def verify_stage(run_id: str) -> int:") != 1:
        parser.error("Unexpected verify_stage structure; refused to edit")

    print(f"TARGET: {target}")
    print(f"SOURCE_SHA256: {sha256(raw)}")
    print("".join(difflib.unified_diff(
        current.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="tests/verify_lossless.py (before)",
        tofile="tests/verify_lossless.py (after)",
    )))
    if not args.apply:
        print("DRY_RUN_ONLY: no files modified; use --apply after reviewing diff")
        return 0

    # Back up untracked source safely: git may not provide an earlier copy.
    backup_dir = root / "experiments/runs/m2_03/diagnostics/code_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"verify_lossless.before_guard.{sha256(raw)}.py.bak"
    if backup_path.exists():
        if backup_path.read_bytes() != raw:
            parser.error(f"Backup content collision; refusing to edit: {backup_path}")
    else:
        with backup_path.open("xb") as handle:
            handle.write(raw)

    # Verify file not modified during review/backup before replacing source.
    if target.read_bytes() != raw:
        parser.error("Source changed during patch preparation; refused to edit")
    target.write_bytes(candidate.encode("utf-8"))
    print(f"BACKUP: {backup_path}")
    print(f"PATCH_APPLIED: PASS sha256={sha256(target.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
