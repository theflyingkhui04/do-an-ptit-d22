"""Regression tests for existing run-ID evidence guard; no GPU/network required.

Run from repo root:
    python -m pytest -q tests/test_verify_lossless_evidence_guard.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Match the project's existing test import convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_lossless as v  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoNetworkSession:
    """Session may be constructed in old buggy code but must never send HTTP."""

    def get(self, *args, **kwargs):
        pytest.fail("No HTTP GET is permitted in duplicate-run rejection")

    def post(self, *args, **kwargs):
        pytest.fail("No HTTP POST is permitted in duplicate-run rejection")

    def close(self):
        pass


@pytest.mark.parametrize(
    "existing_names",
    [
        ("adaptive.jsonl", "summary.json"),
        ("adaptive.jsonl",),
        ("summary.json",),
    ],
    ids=["both-existing", "adaptive-only", "summary-only"],
)
def test_verify_stage_rejects_existing_run_without_modifying_evidence(
    tmp_path: Path, existing_names: tuple[str, ...]
) -> None:
    run_id = "existing-run"
    run_root = tmp_path / "runs"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    canonical = tmp_path / "results" / "m2_lossless_verification_summary.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b'{"old_run":true,"token_id_gate_passed":false}\n')

    for name in existing_names:
        (run_dir / name).write_bytes(
            b'{"source":"old_run","failures":10}\n'
            if name == "summary.json"
            else b'{"sample_id":"frozen_001","token_ids":[1,2,3]}\n'
        )
    protected = [canonical] + [run_dir / name for name in existing_names]
    before = {p: digest(p) for p in protected}

    # On the buggy implementation, load_plan raises inside try/finally, which
    # overwrites both summaries. After patch, guard returns before any call.
    with (
        patch.object(v, "RUN_ROOT", run_root),
        patch.object(v, "SUMMARY_PATH", canonical),
        patch.object(v, "load_plan", side_effect=v.VerificationError("should never load")) as plan,
        patch.object(v.requests, "Session", return_value=NoNetworkSession()) as session,
    ):
        rc = v.verify_stage(run_id)

    assert rc != 0, "Reused run IDs must be rejected"
    assert plan.call_count == 0, "Rejection must precede all plan/runtime work"
    assert session.call_count == 0, "Rejection must precede Session construction"
    assert {p: digest(p) for p in protected} == before, "Evidence bytes changed"
    assert (run_dir / "adaptive.jsonl").exists() == ("adaptive.jsonl" in existing_names)
    assert (run_dir / "summary.json").exists() == ("summary.json" in existing_names)


def test_fresh_run_keeps_original_fail_closed_summary_behavior(tmp_path: Path) -> None:
    """Guard must not suppress the previous failed-run summary behavior."""
    run_root = tmp_path / "runs"
    run_dir = run_root / "new-run"
    run_dir.mkdir(parents=True)
    canonical = tmp_path / "results" / "m2_lossless_verification_summary.json"
    with (
        patch.object(v, "RUN_ROOT", run_root),
        patch.object(v, "SUMMARY_PATH", canonical),
        patch.object(v, "load_plan", side_effect=v.VerificationError("simulated input failure")) as plan,
        patch.object(v.requests, "Session", return_value=NoNetworkSession()),
    ):
        rc = v.verify_stage("new-run")
    assert rc != 0
    assert plan.call_count == 1
    assert v.load_json(canonical)["token_id_gate_passed"] is False
    assert v.load_json(run_dir / "summary.json")["token_id_gate_passed"] is False
