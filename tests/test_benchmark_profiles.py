"""Environment profiles of the benchmark harness: T4 local (default) and L4 remote server."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from experiments import benchmark_harness as harness


L4 = "m3-l4-modal-v1"
T4 = "m1-t4-cu130-v1"

SERVER_ENV = {
    "gpu": "NVIDIA L4",
    "driver_version": "610.57.04",
    "gpu_memory_total_mb": 23034.0,
    "torch_version": "2.13.0+cu130",
    "torch_cuda_version": "13.0",
    "vllm_version": "0.30.0",
    "dtype": "bfloat16",
    "quantization": "awq_marlin",
    "lane": "vanilla",
    "run_id": "r1",
}


@pytest.fixture(autouse=True)
def _server_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PTIT_SERVER_ENV_ROOT", str(tmp_path))


def write_env(tmp_path, **override):
    path = tmp_path / "server_env.json"
    path.write_text(json.dumps({**SERVER_ENV, **override}), encoding="utf-8")
    return path


def parse(argv):
    with patch.object(sys, "argv", ["harness", "--model", "m", *argv]):
        return harness.parse_args()


# ------------------------------------------------------------- defaults
def test_default_profile_and_constants_unchanged():
    assert harness.ENVIRONMENT_PROFILE == T4
    assert harness.EXPECTED_GPU == "Tesla T4"
    assert harness.ENVIRONMENT_PROFILES[T4]["gpu"] == "Tesla T4"
    assert parse([]).environment_profile == T4


def test_l4_profile_expectations():
    assert harness.ENVIRONMENT_PROFILES[L4] == {
        "gpu": "NVIDIA L4",
        "vllm_version": "0.30.0",
        "torch_version": "2.13.0+cu130",
        "torch_cuda_version": "13.0",
        "mode": "server",
    }


def test_l4_profile_requires_server_env_json():
    with pytest.raises(SystemExit):
        parse(["--environment-profile", L4])


def test_help_renders(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse(["--help"])

    assert exit_info.value.code == 0
    assert "cycle-rotate" in capsys.readouterr().out


def test_unknown_profile_rejected():
    with pytest.raises(SystemExit):
        parse(["--environment-profile", "nope"])


# ------------------------------------------------------------- server env
def test_load_server_env_ok(tmp_path):
    assert harness.load_server_env(write_env(tmp_path))["gpu"] == "NVIDIA L4"


def test_load_server_env_missing_fields(tmp_path):
    path = tmp_path / "server_env.json"
    path.write_text(json.dumps({"gpu": "NVIDIA L4"}), encoding="utf-8")

    with pytest.raises(harness.ServerEnvError, match="vllm_version"):
        harness.load_server_env(path)


def test_load_server_env_missing_file(tmp_path):
    with pytest.raises(harness.ServerEnvError, match="not found"):
        harness.load_server_env(tmp_path / "missing.json")


def test_server_environment_does_not_touch_local_gpu(tmp_path):
    with patch.object(harness, "nvmlInit", side_effect=AssertionError("local NVML used")):
        env, handle = harness.environment_info(0, profile=L4, server_env_json=write_env(tmp_path))

    assert handle is None
    assert env["environment_profile"] == L4
    assert env["environment_source"] == "server"
    assert env["gpu"] == "NVIDIA L4"
    assert env["vllm_version"] == "0.30.0"
    assert env["server_lane"] == "vanilla"


def test_validate_l4_profile_reports_gpu_mismatch(tmp_path):
    env = {**SERVER_ENV, "gpu": "NVIDIA A10G"}

    mismatches = harness.validate_environment(env, profile=L4)

    assert len(mismatches) == 1
    assert "gpu" in mismatches[0]


def test_validate_l4_profile_passes():
    assert harness.validate_environment(SERVER_ENV, profile=L4) == []


def test_validate_default_profile_still_t4():
    assert any("gpu" in m for m in harness.validate_environment(SERVER_ENV))


# ------------------------------------------------------------- repeat timing / VRAM
class FakeWriter:
    path = "raw.jsonl"

    def __init__(self):
        self.records = []

    def write(self, records):
        self.records.extend(records)


def run_one_repeat(gpu_handle):
    args = type("A", (), {"num_requests": 2, "vram_sample_interval_s": 0.01})()
    records = [{"success": True, "metrics_complete": True, "output_tokens": 5} for _ in range(2)]
    writer = FakeWriter()
    with patch.object(harness, "run_requests", AsyncMock(return_value=records)):
        code = asyncio.run(harness.run_repeat(
            session=None, args=args, env={}, workload_rows=[{}], workload_integrity=None,
            sample_policy="cycle-rotate", gpu_handle=gpu_handle, raw_writer=writer,
            experiment_id="e", cell_id="c", concurrency=1, repeat_index=1,
        ))
    return code, writer.records


def test_server_mode_repeat_has_utc_window_and_no_local_vram():
    with patch.object(harness, "monitor_vram", side_effect=AssertionError("local VRAM sampled")):
        code, records = run_one_repeat(gpu_handle=None)

    assert code == 0
    for record in records:
        assert record["run_vram_peak_mb"] is None
        start = datetime.fromisoformat(record["run_started_at_utc"])
        end = datetime.fromisoformat(record["run_finished_at_utc"])
        assert start <= end
        assert start.tzinfo is not None


def test_local_mode_repeat_still_samples_vram():
    with patch.object(harness, "monitor_vram", AsyncMock(return_value=123.0)):
        _, records = run_one_repeat(gpu_handle=object())

    assert all(r["run_vram_peak_mb"] == 123.0 for r in records)
    assert all("run_started_at_utc" in r for r in records)


# ------------------------------------------------------------- async_main
def test_main_stops_before_requests_when_server_env_missing(tmp_path):
    args = parse(["--environment-profile", L4, "--server-env-json", str(tmp_path / "missing.json"),
                  "--raw-out", str(tmp_path / "raw.jsonl"), "--summary-csv", str(tmp_path / "s.csv")])

    with patch.object(harness, "run_sweep", side_effect=AssertionError("requests sent")), \
         patch.object(harness, "nvmlShutdown", side_effect=AssertionError("NVML never initialised")):
        assert asyncio.run(harness.async_main(args)) == 5


def test_server_env_outside_allowed_root_rejected(tmp_path, monkeypatch):
    inside = tmp_path / "root"
    inside.mkdir()
    monkeypatch.setenv("PTIT_SERVER_ENV_ROOT", str(inside))

    path = write_env(tmp_path)
    with pytest.raises(harness.ServerEnvError, match="outside"):
        harness.load_server_env(path)
