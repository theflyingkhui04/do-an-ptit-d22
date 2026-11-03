"""M3 PM lock, packaging and fail-closed validation without GPU execution."""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

import pytest

from experiments import benchmark_harness as harness, modal_lanes as lanes
from experiments.bench_source import snapshot
from experiments.run_m3_campaign import plan, save_plan, selected_attempt, validate_campaign
from experiments.validate_m3 import validate_cell
from experiments.workload_adapter import load_workload_rows, verify_workload_integrity

ROOT = Path(__file__).resolve().parents[1]


def test_committed_snapshot_ignores_checkout_bytes_and_has_canonical_workloads():
    packed, sha = snapshot(ROOT)
    try:
        assert len(sha) == 40
        for artifact in lanes.BENCH_WORKLOADS.values():
            path = Path(packed.name) / artifact
            assert b"\r\n" not in path.read_bytes()
            assert verify_workload_integrity(path).actual_sha256
        assert not (Path(packed.name) / "experiments/runs").exists()
    finally:
        packed.cleanup()


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16, 32])
def test_m3_exact_three_warmups_are_recorded(tmp_path, concurrency):
    args = argparse.Namespace(warmup_policy="m3-fixed3", warmup_requests=3, raw_out=tmp_path / "raw.jsonl")
    requests = AsyncMock(return_value=[{"success": True, "phase": "warmup"} for _ in range(3)])
    with patch.object(harness, "run_requests", requests):
        assert asyncio.run(harness.warm_up_cell(
            session=None, args=args, env={}, workload_rows=[], workload_integrity=None,
            sample_policy="cycle-rotate", experiment_id="cpu-test", cell_id="test", concurrency=concurrency)) == 0
    assert requests.call_args.kwargs["count"] == 3
    assert len((tmp_path / "warmup.jsonl").read_text().splitlines()) == 3
    assert not (tmp_path / "raw.jsonl").exists()


def test_failed_warmup_does_not_enter_measurement(tmp_path):
    args = argparse.Namespace(lane="eagle", workload="W1", repeats=5,
                              warmup_policy="m3-fixed3", warmup_requests=3, raw_out=tmp_path / "raw.jsonl")
    warm = [{"success": False, "request_id": str(i), "sample_id": str(i),
             "error_type": "timeout", "error_message": "CPU fixture"} for i in range(3)]
    with patch.object(harness, "run_requests", AsyncMock(return_value=warm)), \
            patch.object(harness, "run_repeat", AsyncMock()) as measured:
        assert asyncio.run(harness.run_cell(
            session=None, args=args, env={}, workload_rows=[], workload_integrity=None,
            sample_policy="cycle-rotate", experiment_id="cpu-test", concurrency=32,
            gpu_handle=None, raw_writer=None)) == 4
        measured.assert_not_called()


@pytest.fixture
def evidence(tmp_path):
    packed, sha = snapshot(ROOT)
    try:
        path = Path(packed.name) / lanes.BENCH_WORKLOADS["W1"]
        integrity = verify_workload_integrity(path)
        corpus = load_workload_rows(path)
    finally:
        packed.cleanup()
    receipt = {"lane": "eagle", "server_run_id": "cpu-server", "bench_run_id": "cpu-cell",
               "call_id": "fc-cpufixture", "commit": sha,
               "cell": {"workload": "W1", "concurrency": 1, "repeats": 5, "num_requests": 32}}
    server = lanes.build_server_env(gpu="NVIDIA L4", driver_version="fixture", gpu_memory_total_mb=23034,
                                   torch_version="fixture", torch_cuda_version="13.0", vllm_version="0.30.0",
                                   lane="eagle", run_id="cpu-server", profile="l4-bf16")
    server.update(source_commit=sha, instance_id="cpu-boot", vllm_args=lanes.build_vllm_args("eagle"))
    raw, warm = [], []
    for repeat in range(6):
        phase = "warmup" if repeat == 0 else "measured"
        offset = ((repeat - 1) * (32 % len(corpus))) % len(corpus) if repeat else 0
        for index in range(3 if repeat == 0 else 32):
            sample_index = (index + offset) % len(corpus)
            request = corpus[sample_index]["request"]
            row = {"experiment_id": "cpu-cell", "cell_id": "eagle__W1__c1", "lane": "eagle", "workload": "W1",
                   "concurrency": 1, "phase": phase, "repeat_id": f"r{repeat:03}" if repeat else None,
                   "request_id": f"cpu-{repeat}-{index}", "request_index": index, "commit": sha,
                   "server_run_id": "cpu-server", "server_instance_id": "cpu-boot", "sample_policy": "cycle-rotate",
                   "sample_id": corpus[sample_index]["sample_id"], "sample_index": sample_index,
                   "sample_offset": offset, "canonical_sample_count": len(corpus), "success": True,
                   "workload_file_sha256": integrity.actual_sha256, "metrics_complete": True, "http_status": 200,
                   "target_revision": lanes.TARGET_REVISION, "draft_revision": lanes.EAGLE_REVISION,
                   "model": lanes.EAGLE_SERVED_MODEL, "temperature": request["temperature"], "seed": request["seed"],
                   "max_completion_tokens": request["max_tokens"], "response_format": request.get("response_format"),
                   "timestamp_utc": f"2026-11-03T00:0{repeat}:00+00:00", "e2e_ms": 1000,
                   "ttft_ms": 10, "tpot_ms": 20, "output_tokens": 50, "gpu": "NVIDIA L4",
                   "run_id": f"cpu-run-{repeat}", "run_duration_s": 32,
                   "server_metrics": {"speculative_decoding": {"num_spec_steps": 5, "num_draft_tokens": 15,
                                       "num_accepted_draft_tokens": 5, "num_spec_tokens": 3}}}
            (raw if repeat else warm).append(row)
    def write(name, value):
        (tmp_path / name).write_bytes((json.dumps(value) + "\n").encode())
    write("submission.json", receipt)
    write("bench.json", {**receipt, "returncode": 0, "server_instance_unchanged": True,
                         "endpoint_model_ids": [lanes.EAGLE_SERVED_MODEL]})
    write("server_env.json", server)
    (tmp_path / "raw.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in raw).encode())
    (tmp_path / "warmup.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in warm).encode())
    return tmp_path


def test_full_cell_validator_accepts_real_contract(evidence):
    result = validate_cell(evidence)
    assert result["measured_requests"] == 160
    assert result["warmup_successful"] == 3
    assert result["speculative_alpha_tau"] == (1 / 3, 2)


@pytest.mark.parametrize("field,value", [("commit", "a" * 40), ("sample_id", "wrong"), ("phase", "warmup"),
    ("success", False), ("seed", 99), ("draft_revision", "wrong"), ("server_instance_id", "old-server")])
def test_validator_rejects_corrupt_measured_evidence(evidence, field, value):
    path = evidence / "raw.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode())
    with pytest.raises(ValueError):
        validate_cell(evidence)


def test_validator_rejects_missing_request_and_extra_warmup(evidence):
    path = evidence / "raw.jsonl"
    path.write_bytes("\n".join(path.read_text().splitlines()[1:]).encode())
    with pytest.raises(ValueError, match="160"):
        validate_cell(evidence)


def test_frozen_grid_and_no_duplicate_completed_cells(tmp_path, monkeypatch):
    from experiments import run_m3_campaign as campaign
    record = plan("cpu-confirmatory", "a" * 40)
    assert len(record["cells"]) == 72
    assert len({c["key"] for c in record["cells"]}) == 72
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    path = save_plan(record)
    assert save_plan(record) == path
    with pytest.raises(FileExistsError):
        save_plan({**record, "repeats": 4})
    with pytest.raises(ValueError, match="Incomplete campaign"):
        validate_campaign(record)
    monkeypatch.setattr(campaign, "attempts", lambda r, c: [tmp_path / "a", tmp_path / "b"])
    for name in ("a", "b"):
        out = tmp_path / name
        out.mkdir()
        (out / "bench.json").write_text("{}")
        (out / "submission.json").write_text(json.dumps({"lane": "vanilla", "cell": {"workload": "W1", "concurrency": 1}}))
    monkeypatch.setattr(campaign, "validate_cell", lambda p: {"commit": "a" * 40})
    with pytest.raises(ValueError, match="Duplicate complete"):
        selected_attempt(record, record["cells"][0])


def test_resume_does_not_submit_or_stop_an_existing_live_call(tmp_path, monkeypatch):
    import experiments
    from experiments import run_m3_campaign as campaign
    record = plan("cpu-confirmatory", "a" * 40)
    record["cells"] = record["cells"][:1]
    pending = tmp_path / "existing-attempt"
    runner = SimpleNamespace(_committed_bench_source=lambda: "a" * 40,
                             bench_status=Mock(side_effect=["RUNNING", ConnectionError("laptop offline")]),
                             bench_submit=Mock(), bench_collect=Mock())
    monkeypatch.setattr(experiments, "modal_runner", runner, raising=False)
    monkeypatch.setattr(campaign, "save_plan", Mock())
    monkeypatch.setattr(campaign, "selected_attempt", lambda r, c: None)
    monkeypatch.setattr(campaign, "attempts", lambda r, c: [pending])
    shutdown = Mock()
    monkeypatch.setattr(campaign, "stop", shutdown)
    with pytest.raises(ConnectionError):
        campaign.drive(record)
    runner.bench_submit.assert_not_called()
    shutdown.assert_not_called()


def test_resume_skips_collected_cells_without_remote_operations(monkeypatch):
    import experiments
    from experiments import run_m3_campaign as campaign
    runner = SimpleNamespace(_committed_bench_source=lambda: "a" * 40, bench_submit=Mock(), bench_status=Mock())
    monkeypatch.setattr(experiments, "modal_runner", runner, raising=False)
    monkeypatch.setattr(campaign, "save_plan", Mock())
    monkeypatch.setattr(campaign, "selected_attempt", lambda r, c: Path("valid-attempt"))
    shutdown = Mock()
    monkeypatch.setattr(campaign, "stop", shutdown)
    campaign.drive(plan("cpu-confirmatory", "a" * 40))
    runner.bench_submit.assert_not_called()
    runner.bench_status.assert_not_called()
    shutdown.assert_not_called()
