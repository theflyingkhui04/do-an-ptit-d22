"""Tests for Modal cloud serving runner (BYOC, ephemeral, and deploy modes)."""
import os
from unittest.mock import patch
import pytest
from flexsds.server.modal_runner import ModalServingRunner


def test_modal_runner_build_command_ephemeral_and_deploy():
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "L4", "port": 8000}}
    cmd_run = runner.build_modal_command(config, deploy=False)
    assert "modal" in cmd_run[0]
    assert "run" in cmd_run

    cmd_deploy = runner.build_modal_command(config, deploy=True)
    assert "deploy" in cmd_deploy


def test_modal_runner_credentials_check_env(monkeypatch):
    runner = ModalServingRunner()
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    assert runner.check_credentials() is True


def test_modal_runner_credentials_check_missing(monkeypatch):
    runner = ModalServingRunner()
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with patch("os.path.exists", return_value=False):
        assert runner.check_credentials() is False


def test_modal_runner_stop_command():
    runner = ModalServingRunner()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        success = runner.stop(app_name="ptit-vllm")
        assert success is True
        mock_run.assert_called_once()
