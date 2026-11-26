"""End-to-End integration tests for FlexSDS CLI across all 3 platforms and end-to-end user workflows."""
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from flexsds.cli.main import main
from flexsds.cli.wizard import run_interactive_wizard
from flexsds.core.hardware import HardwareInfo
from flexsds.core.credentials import CredentialsManager
from flexsds.client.client import FlexSDSClient
from flexsds.client.chat import InteractiveChatSession
from flexsds.server.modal_runner import ModalServingRunner


def test_e2e_cli_local_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "local",
        "--port", "8000",
        "--dry-run",
        "--allow-fallback",
        "--no-adaptive",  # adaptive needs a measured GPU profile; these check the command shape
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "vllm serve" in captured.out
    assert "--speculative-config" in captured.out
    assert "AdaptiveAsyncScheduler" not in captured.out
    assert "--scheduler-cls" not in captured.out


def test_e2e_cli_modal_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "modal",
        "--dry-run",
        "--allow-fallback",
        "--no-adaptive",  # adaptive needs a measured GPU profile; these check the command shape
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    # The runner launches the resolved binary (shutil.which), e.g. ...\Scripts\modal.EXE on Windows
    assert re.search(r"modal(\.exe|\.cmd)?\s+serve\b", captured.out, re.IGNORECASE)
    assert "modal_app.py" in captured.out


def test_e2e_cli_modal_deploy_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "modal",
        "--deploy",
        "--dry-run",
        "--allow-fallback",
        "--no-adaptive",  # adaptive needs a measured GPU profile; these check the command shape
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert re.search(r"modal(\.exe|\.cmd)?\s+deploy\b", captured.out, re.IGNORECASE)
    assert "modal_app.py" in captured.out


def test_e2e_cli_colab_dry_run(capsys):
    test_args = [
        "serve",
        "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "--backend", "colab",
        "--dry-run",
        "--allow-fallback",
        "--no-adaptive",  # adaptive needs a measured GPU profile; these check the command shape
    ]
    ret = main(test_args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Colab guidance only" in captured.out


def test_e2e_cli_detect_command(capsys):
    ret = main(["detect"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Hardware:" in captured.out


def test_e2e_cli_profile_command(capsys):
    ret = main(["profile"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Linked hardware profile:" in captured.out


def test_e2e_cli_stop_command(capsys):
    ret = main(["stop"])
    assert ret == 0


def test_end_to_end_wizard_generation_and_guidance(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    hw = HardwareInfo(
        device_name="NVIDIA L4",
        total_vram_mb=24000.0,
        free_vram_mb=20000.0,
        sm_architecture="sm_89",
        cuda_version="12.4",
        is_cuda_available=True,
    )

    monkeypatch.setattr("flexsds.cli.wizard.prompt_choice", lambda q, c, **kw: c[0])
    monkeypatch.setattr("flexsds.cli.wizard.prompt_string", lambda q, d: d)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_bool", lambda q, **kw: False)

    cfg = run_interactive_wizard(hw)

    assert cfg["action"] == "serve"
    assert "target_model" in cfg
    assert "draft_model" in cfg

    saved_file = tmp_path / "serving_config.json"
    assert saved_file.exists()
    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["action"] == "serve"

    captured = capsys.readouterr()
    assert "flexsds chat" in captured.out


def test_end_to_end_auth_and_modal_runner(tmp_path, monkeypatch):
    cred_file = tmp_path / "credentials"
    monkeypatch.setattr("flexsds.core.credentials.CredentialsManager.DEFAULT_PATH", cred_file)

    mgr = CredentialsManager()
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-test-token-id")
    mgr.set_credential("modal", "MODAL_TOKEN_SECRET", "as-test-token-secret")

    runner = ModalServingRunner()
    assert runner.check_credentials() is True

    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "target_gpu": "A10",
        }
    }
    cmd = runner.build_modal_command(cfg, deploy=False)
    assert Path(cmd[0]).stem.lower() == "modal"
    assert "modal_app.py" in cmd[2]
    assert json.loads(runner.setup_environment(cfg)["FLEXSDS_SERVING_CONFIG"])["target_gpu"] == "A10"


def test_end_to_end_chat_session_with_client_stream(monkeypatch, capsys):
    mock_client = MagicMock(spec=FlexSDSClient)
    mock_client.base_url = "http://localhost:8000/v1"
    mock_client.check_health.return_value = True
    mock_client.stream_chat.return_value = iter(["FlexSDS", " phục", " vụ", " thành", " công!"])

    session = InteractiveChatSession(mock_client, model="qwen2.5-7b-adaptive")

    inputs = iter(["Thử nghiệm hệ thống", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    code = session.start()
    assert code == 0

    captured = capsys.readouterr()
    assert "FlexSDS phục vụ thành công!" in captured.out
    assert "Inference Telemetry" in captured.out or "Thống kê suy luận" in captured.out
    assert "SSE content chunks" in captured.out
