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


def test_local_runner_stop_process_group():
    runner = LocalServingRunner()
    pid_file = "/tmp/test_flexsds.pid"
    with open(pid_file, "w") as f:
        f.write("99999\n")

    with patch("os.killpg") as mock_killpg, patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
        stopped = runner.stop(pid_file=pid_file)
        assert stopped is True
        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)
        mock_remove.assert_called_once_with(pid_file)
