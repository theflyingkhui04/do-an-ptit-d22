"""Executed CLI contract checks, not a model routing/compliance evaluator."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from flexsds.cli import main as cli
from flexsds.core.hardware import HardwareInfo
from flexsds.core.cost_model import load_profile_for_hardware
from flexsds.server.local_runner import LocalServingRunner
from flexsds.server.modal_runner import ModalServingRunner


ROOT = Path(__file__).resolve().parents[1]
PAIR = ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "Qwen/Qwen2.5-0.5B-Instruct"]


def invoke(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "flexsds.cli.main", *args],
                          cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, encoding="utf-8", timeout=45)


@pytest.mark.parametrize("args, expected", [
    (["--version"], "0.2.0-alpha"),
    (["--help"], "{detect,profile,stop,auth,serve,chat,monitor}"),
    (["detect"], "Hardware:"),
    (["profile"], "Linked hardware profile:"),
    (PAIR + ["--backend", "local", "--dry-run"], "Dry-run execution command:"),
    (PAIR + ["--backend", "modal", "--dry-run"], "Dry-run command"),
])
def test_real_safe_cli(args, expected, tmp_path):
    result = invoke(args, tmp_path)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("args", [["test"], ["check-compat"], ["detect", "--json"],
                                 PAIR + ["--port", "invalid"]])
def test_real_controlled_parser_failure(args, tmp_path):
    result = invoke(args, tmp_path)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_executable_is_observable(tmp_path):
    with pytest.raises(FileNotFoundError):
        subprocess.run([str(tmp_path / "missing-flexsds-executable")], check=True)


@pytest.fixture
def cpu_cli(monkeypatch):
    hw = HardwareInfo("CPU", "cpu", 16000, 16000, "N/A", False)
    monkeypatch.setattr(cli, "detect_hardware", lambda: hw)
    monkeypatch.setattr(cli, "print_banner", lambda hw: None)
    return hw


def test_cpu_profile_is_l4_fallback(cpu_cli, capsys):
    assert cli.main(["detect"]) == 0
    assert "CPU Only" in capsys.readouterr().out
    assert load_profile_for_hardware(cpu_cli)[1] == "profile_sm89_l4.json"


@pytest.mark.parametrize("args", [[], ["serve", "--dry-run"],
                                 ["serve", "--target", "x", "--draft", "", "--dry-run"]])
def test_incomplete_input_enters_wizard_even_dry_run(args, cpu_cli, monkeypatch):
    wizard = Mock()
    monkeypatch.setattr(cli, "run_interactive_wizard", wizard)
    assert cli.main(args) == 0
    wizard.assert_called_once_with(cpu_cli)


@pytest.mark.parametrize("backend", ["local", "modal"])
def test_dry_run_never_calls_popen(backend, cpu_cli, monkeypatch, capsys):
    popen = Mock(side_effect=AssertionError("Unexpected server start"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    assert cli.main(PAIR + ["--backend", backend, "--dry-run"]) == 0
    assert "Dry-run" in capsys.readouterr().out
    popen.assert_not_called()


def test_oom_warning_still_returns_zero(cpu_cli, monkeypatch, capsys):
    cpu_cli.free_vram_mb = 1
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("launch")))
    assert cli.main(PAIR + ["--backend", "local", "--dry-run"]) == 0
    assert "WARNING/INCOMPATIBLE" in capsys.readouterr().out


def test_missing_runner_binary_returns_nonzero_with_error(cpu_cli, monkeypatch, capsys):
    cpu_cli.is_cuda_available = True
    # Controlled stand-in only: no actual serving process can be created.
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=FileNotFoundError))
    assert cli.main(PAIR + ["--backend", "local"]) == 1
    assert "Server binary not found" in capsys.readouterr().out


def test_no_adaptive_uses_default_scheduler_and_executor():
    runner = LocalServingRunner()
    cmd = runner.build_command({"enable_adaptive": False})
    assert "--speculative-config" in cmd
    assert "--scheduler-cls" not in cmd
    assert "--distributed-executor-backend" not in cmd
    assert "num_speculative_tokens_per_batch_size" not in " ".join(cmd)


def test_modal_forwards_models_port_and_gpu_in_environment():
    import json
    runner = ModalServingRunner()
    cfg = {"target_model": "chosen-target", "draft_model": "chosen-draft",
           "port": 9876, "target_gpu": "T4", "enable_adaptive": False}
    command = runner.build_modal_command(cfg)
    assert command[:2] == ["modal", "serve"]
    assert command[2].endswith("modal_app.py")
    assert json.loads(runner.setup_environment(cfg)["FLEXSDS_SERVING_CONFIG"]) == cfg
