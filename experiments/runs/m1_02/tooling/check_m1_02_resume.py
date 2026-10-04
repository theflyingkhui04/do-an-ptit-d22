#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


FROZEN_SHA = "f307e33f9aea4bbd45499f548480cb7d6f90a4b9"
VALIDATOR = Path("/kaggle/working/validate_m1_02_sweep.py")


def fail(message: str) -> None:
    raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether one M1-02 sweep is safe to resume/skip."
    )

    parser.add_argument(
        "--evidence-repo",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--lane",
        required=True,
        choices=["vanilla", "static"],
    )

    parser.add_argument(
        "--workload",
        required=True,
        choices=["W1", "W2", "W3"],
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()

    repo = args.evidence_repo.resolve()

    if not (repo / ".git").exists():
        fail(f"not a Git checkout: {repo}")

    workload_lower = args.workload.lower()

    rel_dir = (
        Path("experiments/runs/m1_02")
        / args.experiment_id
    )

    raw_rel = rel_dir / f"{args.lane}_{workload_lower}.jsonl"
    done_rel = rel_dir / f"{args.lane}_{workload_lower}.done.json"

    raw = repo / raw_rel
    done = repo / done_rel

    print("=== M1-02 RESUME CHECK ===")
    print(f"experiment_id : {args.experiment_id}")
    print(f"lane          : {args.lane}")
    print(f"workload      : {args.workload}")
    print(f"raw           : {raw_rel}")
    print(f"marker        : {done_rel}")

    raw_exists = raw.is_file()
    done_exists = done.is_file()

    print(f"raw exists    : {raw_exists}")
    print(f"marker exists : {done_exists}")

    # Neither artifact exists: this sweep has never completed.
    if not raw_exists and not done_exists:
        print()
        print("RESUME STATE: NOT_STARTED")
        print("ACTION: RUN WHOLE SWEEP")
        return 10

    # One without the other is never a completed checkpoint.
    if raw_exists != done_exists:
        fail(
            "partial checkpoint: raw and done marker "
            "must either both exist or both be absent"
        )

    try:
        marker = json.loads(done.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid done marker JSON: {exc}")

    if not isinstance(marker, dict):
        fail("done marker must be a JSON object")

    expected_marker = {
        "schema": "m1_02_sweep_done.v1",
        "experiment_id": args.experiment_id,
        "lane": args.lane,
        "workload": args.workload,
        "benchmark_commit": FROZEN_SHA,
        "raw_path": raw_rel.as_posix(),
        "measured_records": 960,
        "expected_measured_records": 960,
        "validation_status": "PASS",
    }

    for field, expected in expected_marker.items():
        actual = marker.get(field)

        if actual != expected:
            fail(
                f"marker {field}: expected "
                f"{expected!r}, got {actual!r}"
            )

    marker_sha = marker.get("raw_sha256")

    if (
        not isinstance(marker_sha, str)
        or len(marker_sha) != 64
    ):
        fail(
            f"invalid marker raw_sha256: {marker_sha!r}"
        )

    actual_sha = sha256(raw)

    print(f"marker SHA256 : {marker_sha}")
    print(f"actual SHA256 : {actual_sha}")

    if actual_sha != marker_sha:
        fail(
            "raw SHA256 does not match done marker"
        )

    line_count = sum(
        1
        for line in raw.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    print(f"measured lines: {line_count}")

    if line_count != 960:
        fail(
            f"expected 960 raw records, got {line_count}"
        )

    print()
    print("=== RE-RUN CANONICAL VALIDATOR ===")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--raw",
            str(raw),
            "--lane",
            args.lane,
            "--workload",
            args.workload,
            "--experiment-id",
            args.experiment_id,
        ],
        check=False,
    )

    if completed.returncode != 0:
        fail(
            "canonical validator rejected supposedly "
            f"completed sweep; rc={completed.returncode}"
        )

    # Ensure both files are actually committed in current HEAD.
    for rel in (raw_rel, done_rel):
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "cat-file",
                "-e",
                f"HEAD:{rel.as_posix()}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode != 0:
            fail(
                f"artifact is not present in evidence HEAD: {rel}"
            )

    print()
    print("========================================")
    print("RESUME STATE: COMPLETE_VALID")
    print("ACTION: SKIP THIS SWEEP")
    print("========================================")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"RESUME STATE: INVALID: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
