"""Tests for Modal cloud serving runner (BYOC, ephemeral, and deploy modes)."""
import os
import shutil
from unittest.mock import patch
import pytest
from flexsds.server.modal_runner import ModalServingRunner


def test_modal_runner_build_command_ephemeral_and_deploy():
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "L4", "port": 8000}}
    cmd_run = runner.build_modal_command(config, deploy=False)
    assert os.path.basename(cmd_run[0]) == "modal"
    assert "serve" in cmd_run

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


def test_modal_runner_build_command_uses_standalone_app():
    runner = ModalServingRunner()
    cfg = {"serving": {"target_gpu": "A10G"}}
    cmd = runner.build_modal_command(cfg, deploy=False)
    assert os.path.basename(cmd[0]) == "modal"
    assert "serve" in cmd
    assert any("modal_app.py" in arg for arg in cmd)
    assert '"target_gpu": "A10G"' in runner.setup_environment(cfg)["FLEXSDS_SERVING_CONFIG"]


def test_modal_runner_loads_credentials_from_manager(tmp_path, monkeypatch):
    from flexsds.core.credentials import CredentialsManager
    cred_file = tmp_path / ".flexsds" / "credentials"
    mgr = CredentialsManager(str(cred_file))
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-credmgr")
    mgr.set_credential("modal", "MODAL_TOKEN_SECRET", "as-credmgr")

    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    real_exists = os.path.exists
    monkeypatch.setattr("os.path.exists", lambda path: False if ".modal.toml" in str(path) else real_exists(path))

    runner = ModalServingRunner(credentials_manager=mgr)
    assert runner.check_credentials() is True


def test_modal_runner_check_process_status():
    from unittest.mock import MagicMock
    runner = ModalServingRunner()

    mock_alive = MagicMock()
    mock_alive.poll.return_value = None
    is_alive, code = runner.check_process_status(mock_alive)
    assert is_alive is True
    assert code is None

    mock_stopped = MagicMock()
    mock_stopped.poll.return_value = 1
    is_alive, code = runner.check_process_status(mock_stopped)
    assert is_alive is False
    assert code == 1


def test_modal_runner_resolves_modal_binary(monkeypatch):
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "A10G"}}

    # Giả lập trên Windows tìm thấy modal.cmd
    monkeypatch.setattr("shutil.which", lambda name: "C:\\Python\\Scripts\\modal.cmd" if name == "modal" else None)
    cmd = runner.build_modal_command(config, deploy=False)
    assert cmd[0] == "C:\\Python\\Scripts\\modal.cmd"
    assert "serve" in cmd


def test_modal_runner_setup_environment_uses_os_pathsep(monkeypatch):
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "A10G"}}
    monkeypatch.setenv("PYTHONPATH", "custom_path")
    env = runner.setup_environment(config)
    pp = env.get("PYTHONPATH", "")
    assert len(pp) > 0
    assert os.pathsep in pp
    assert "custom_path" not in pp


@pytest.fixture(autouse=True)
def isolated_default_credentials(tmp_path, monkeypatch):
    from flexsds.core.credentials import CredentialsManager
    monkeypatch.setattr(CredentialsManager, "DEFAULT_PATH", tmp_path / "no_credentials")


def test_modal_environment_does_not_leak_host_python_packages(monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.setenv('PYTHONPATH','/host/site-packages:/host/dist-packages')
    monkeypatch.setattr(sys,'path',[*sys.path,'/host/venv/site-packages'])
    paths=ModalServingRunner().setup_environment({})['PYTHONPATH'].split(os.pathsep)
    root=Path(__file__).resolve().parents[1]
    assert paths == [str(root/'src'),str(root)]
