"""Tests for local serving runner and zero-zombie process management."""
import os
import signal
from unittest.mock import patch, MagicMock
import pytest
from flexsds.server.local_runner import LocalServingRunner


def test_local_runner_build_command_speculative_json():
    runner = LocalServingRunner()
    config = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "port": 8000,
        }
    }
    cmd = runner.build_command(config)
    assert "vllm" in cmd[0] or "python" in cmd[0]
    assert "--speculative-config" in cmd
    spec_idx = cmd.index("--speculative-config") + 1
    assert "num_speculative_tokens_per_batch_size" in cmd[spec_idx]
    assert "--scheduler-cls" in cmd
    assert "src.server.adaptive_scheduler.AdaptiveAsyncScheduler" in cmd


def test_local_runner_environment_vars():
    runner = LocalServingRunner()
    env = runner.setup_environment({"serving": {"enable_adaptive": True}})
    assert env.get("ADAPTIVE_ENABLE_VRAM_GUARD") == "1"
    assert "ADAPTIVE_TELEMETRY_PATH" in env


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups unavailable on Windows")
def test_local_runner_stop_process_group(tmp_path):
    runner = LocalServingRunner()
    pid_file = str(tmp_path / "test_flexsds.pid")
    with open(pid_file, "w") as f:
        f.write("99999\n")

    with patch("os.killpg") as mock_killpg, patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
        stopped = runner.stop(pid_file=pid_file)
        assert stopped is True
        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)
        mock_remove.assert_called_once_with(pid_file)


def test_local_runner_build_command_contains_lineage_executor():
    runner = LocalServingRunner()
    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "port": 8000,
            "enable_adaptive": True,
        }
    }
    cmd = runner.build_command(cfg)
    assert "--distributed-executor-backend" in cmd
    idx = cmd.index("--distributed-executor-backend")
    assert cmd[idx + 1] == "src.server.lineage_executor.LineageUniProcExecutor"
    assert "--scheduler-cls" in cmd
    idx_s = cmd.index("--scheduler-cls")
    assert cmd[idx_s + 1] == "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"


def test_local_runner_detects_port_conflict(monkeypatch):
    import socket
    runner = LocalServingRunner()
    def mock_socket_bind(self, address):
        raise OSError("Address already in use")
    monkeypatch.setattr(socket.socket, "bind", mock_socket_bind)
    assert runner.check_port_available(8000) is False


def test_local_runner_launch_aborts_on_port_conflict(monkeypatch):
    runner = LocalServingRunner()
    monkeypatch.setattr(runner, "check_port_available", lambda port: False)
    proc = runner.launch({"serving": {"port": 8000}}, dry_run=False)
    assert proc is None
