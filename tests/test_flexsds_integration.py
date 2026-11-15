"""CPU-only safety and command-contract regressions; no serving or cloud calls."""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flexsds.cli import main as cli, wizard
from flexsds.core import compatibility as compat
from flexsds.core.hardware import HardwareInfo
from flexsds.server.local_runner import LocalServingRunner
from flexsds.server.modal_runner import ModalServingRunner

PAIR = ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--no-adaptive"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def machine(monkeypatch):
    hw = HardwareInfo("Mock GPU", "sm_89", 24000, 23000, "test", True)
    monkeypatch.setattr(cli, "detect_hardware", lambda: hw)
    monkeypatch.setattr(cli, "print_banner", lambda hw: None)
    return hw


@pytest.mark.parametrize("backend", ["local", "modal"])
@pytest.mark.parametrize("failure", ["oom", "incompatible", "metadata"])
def test_unsafe_real_serve_never_reaches_runner(machine, monkeypatch, backend, failure):
    if failure == "oom":
        args = ["serve", "--target", "Qwen/Qwen2.5-72B-Instruct", "--draft", PAIR[4]]
    else:
        args = PAIR
        if failure == "incompatible":
            original = compat.check_compatibility
            def incompatible(*a, **kw):
                report = original(*a, **kw)
                report.is_compatible = False
                return report
            monkeypatch.setattr(compat, "check_compatibility", incompatible)
        else:
            from flexsds.core.model_inspection import ModelFiles, InspectionError
            def unavailable(*a, **kw):
                raise InspectionError("Missing model metadata")
            monkeypatch.setattr(ModelFiles, 'config', unavailable)
    local = Mock(side_effect=AssertionError("unsafe local launch"))
    modal = Mock(side_effect=AssertionError("unsafe deployment"))
    monkeypatch.setattr(LocalServingRunner, "launch", local)
    monkeypatch.setattr(ModalServingRunner, "launch", modal)
    assert cli.main(args + ["--backend", backend]) == 2
    local.assert_not_called()
    modal.assert_not_called()


def test_unsafe_preview_is_zero_and_never_starts_process(machine, monkeypatch, capsys):
    machine.free_vram_mb = 1
    popen = Mock(side_effect=AssertionError("preview launched"))
    monkeypatch.setattr("subprocess.Popen", popen)
    assert cli.main(PAIR + ["--backend", "local", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Launch blocked" in output
    assert "Dry-run execution command" in output
    assert "not a readiness check" in output
    popen.assert_not_called()


@pytest.mark.parametrize("backend", ["local", "modal"])
@pytest.mark.parametrize("outcome,expected", [(None, 1), (0, 0), (7, 7)])
def test_safe_dispatch_propagates_launch_result(machine, monkeypatch, backend, outcome, expected):
    proc = None if outcome is None else Mock()
    if proc is not None:
        proc.wait.return_value = outcome
    launch = Mock(return_value=proc)
    runner = LocalServingRunner if backend == "local" else ModalServingRunner
    monkeypatch.setattr(runner, "launch", launch)
    assert cli.main(PAIR + ["--backend", backend]) == expected
    launch.assert_called_once()


@pytest.mark.parametrize("failure", ["oom", "metadata", "incompatible"])
def test_wizard_guard_blocks_before_config_write_and_launch(machine, monkeypatch, tmp_path, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard, "prompt_choice", lambda q, c, **kw:
                        next((v for v in c if "Local vLLM" in v), c[0]))
    if failure == "oom":
        machine.free_vram_mb = 1
    elif failure == "metadata":
        from flexsds.core.model_inspection import ModelFiles, InspectionError
        def unavailable(*a, **kw):
            raise InspectionError("Missing model metadata")
        monkeypatch.setattr(ModelFiles, 'config', unavailable)
    else:
        original = compat.check_compatibility
        def incompatible(*a, **kw):
            report = original(*a, **kw)
            report.is_compatible = False
            return report
        monkeypatch.setattr(compat, "check_compatibility", incompatible)
    launch = Mock(side_effect=AssertionError("unsafe wizard launch"))
    monkeypatch.setattr(LocalServingRunner, "launch", launch)
    with pytest.raises(compat.ServingBlockedError):
        wizard.run_interactive_wizard(machine)
    launch.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_wizard_safe_runner_failure_is_nonzero(machine, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard, "prompt_choice", lambda q, c, **kw:
                        next((v for v in c if "Local vLLM" in v), c[0]))
    monkeypatch.setattr(wizard, "prompt_bool", lambda q, **kw: "background" not in q and "hardwareadaptive" not in q.lower())
    launch = Mock(return_value=None)
    monkeypatch.setattr(LocalServingRunner, "launch", launch)
    assert cli.main([]) == 1
    launch.assert_called_once()


def test_port_conflict_never_starts_process(monkeypatch):
    runner = LocalServingRunner()
    monkeypatch.setattr(runner, "check_port_available", lambda p: False)
    popen = Mock(side_effect=AssertionError("occupied port launched"))
    monkeypatch.setattr("subprocess.Popen", popen)
    assert runner.launch({}) is None
    popen.assert_not_called()


def test_modal_missing_credentials_blocks_and_preview_never_reads_secrets(monkeypatch):
    runner = ModalServingRunner()
    credentials = Mock(return_value=False)
    popen = Mock(side_effect=AssertionError("unauthenticated deploy"))
    monkeypatch.setattr(runner, "check_credentials", credentials)
    monkeypatch.setattr("subprocess.Popen", popen)
    runner.launch({}, dry_run=True)
    credentials.assert_not_called()
    assert runner.launch({}) is None
    popen.assert_not_called()


def test_colab_is_explicit_guidance_without_runner(machine, monkeypatch, capsys):
    popen = Mock(side_effect=AssertionError("Colab launch"))
    monkeypatch.setattr("subprocess.Popen", popen)
    assert cli.main(PAIR + ["--backend", "colab"]) == 0
    assert "guidance only" in capsys.readouterr().out
    popen.assert_not_called()


def test_remote_budget_does_not_use_laptop_ram_or_vram(machine):
    machine.free_vram_mb = 1
    machine.is_cuda_available = False
    preflight = compat.evaluate_serving_preflight(PAIR[2], PAIR[4], machine, "modal", "L4")
    assert preflight.report.free_vram_mb == 23000
    assert preflight.is_safe


def test_no_adaptive_flag_reaches_generated_command(machine, monkeypatch, capsys):
    popen = Mock(side_effect=AssertionError("preview launched"))
    monkeypatch.setattr("subprocess.Popen", popen)
    assert cli.main(PAIR + ["--backend", "local", "--no-adaptive", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "--speculative-config" in output
    assert "--scheduler-cls" not in output
    assert "--distributed-executor-backend" not in output
    popen.assert_not_called()


def test_modal_app_allocates_gpu_exposes_port_and_preserves_exact_config(monkeypatch):
    cfg = {"target_model": PAIR[2], "draft_model": PAIR[4], "draft_method": "draft_model",
           "port": 8765, "target_gpu": "T4", "enable_adaptive": False}
    runner = ModalServingRunner()
    monkeypatch.setenv("FLEXSDS_SERVING_CONFIG", runner.setup_environment(cfg)["FLEXSDS_SERVING_CONFIG"])
    image = Mock()
    for method in ["pip_install", "uv_pip_install", "entrypoint", "env", "add_local_dir", "add_local_file"]:
        getattr(image, method).return_value = image
    app = Mock()
    app.function.side_effect = lambda **kw: lambda f: f
    web = Mock(side_effect=lambda *a, **kw: lambda f: f)
    fake_modal = SimpleNamespace(App=Mock(return_value=app),
        Image=SimpleNamespace(from_registry=Mock(return_value=image)),
        Volume=SimpleNamespace(from_name=Mock(return_value="volume")), web_server=web,
        is_local=lambda: True)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    spec = importlib.util.spec_from_file_location("audit_modal_app", ROOT / "src/flexsds/server/modal_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert app.function.call_args.kwargs["gpu"] == "T4"
    web.assert_called_once_with(8765, startup_timeout=600)
    popen = Mock()
    monkeypatch.setattr("subprocess.Popen", popen)
    module.serve()
    command = popen.call_args.args[0]
    assert command[2] == PAIR[2]
    assert command[command.index("--port") + 1] == "8765"
    assert json.loads(command[command.index("--speculative-config") + 1])["model"] == PAIR[4]
    assert "--scheduler-cls" not in command
    assert runner.build_modal_command(cfg, deploy=True)[1] == "deploy"


def test_imported_helpers_contain_no_embedded_credentials_or_saved_output():
    pattern = re.compile(r"(?:gh[pousr]_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
    # vscode_cli.ipy holds IPython cells (`!git`, `!pip`): not importable Python, so not a .py module
    for path in [ROOT / "src/flexsds/server/vscode_cli.ipy",
                 ROOT / "src/flexsds/server/vscode_cli.ipynb",
                 ROOT / "logs/Khoi/M1-01/vscode_cli.ipynb"]:
        text = path.read_text(encoding="utf-8")
        has_secret = bool(pattern.search(text))
        assert has_secret is False, f"Embedded credential in {path.name} (value withheld)"
        assert "https://{GITHUB_PAT}@" not in text
        if path.suffix == ".ipynb" and "server" in path.parts:
            assert all(not c.get("outputs") for c in json.loads(text)["cells"])


@pytest.mark.parametrize("args", [["serve", "--port", "0"], ["monitor", "--refresh-rate", "0"]])
def test_invalid_ranges_fail_before_side_effects(machine, args):
    assert cli.main(args) == 2


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
def test_new_command_output_encodes_without_stream_reconfiguration(machine, monkeypatch, tmp_path, encoding):
    from test_cli_output_encoding import EncodedStream
    from flexsds.core.credentials import CredentialsManager
    from flexsds.client.chat import InteractiveChatSession
    from flexsds.client.monitor import run_live_monitor
    from flexsds.cli.task_runner import TaskRunner
    sink = EncodedStream(encoding)
    monkeypatch.setattr(sys, "stdout", sink)
    monkeypatch.setattr(CredentialsManager, "DEFAULT_PATH", tmp_path / "credentials")
    assert cli.main(["auth", "status"]) == 0
    client = Mock()
    client.base_url = "http://mock.invalid/v1"
    client.check_health.return_value = True
    client.stream_chat.return_value = iter(["Hello", "世界"])
    inputs = iter(["question", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert InteractiveChatSession(client).start() == 0
    monkeypatch.setattr("flexsds.client.monitor.time.sleep", Mock(side_effect=KeyboardInterrupt))
    assert run_live_monitor(str(tmp_path / "missing")) == 0
    task_runner = TaskRunner(interactive=False)
    task_runner.run_task(task_runner.add_task("check ✓"), lambda t: True)
    task_runner.render_view().encode(encoding)
    runner = ModalServingRunner()
    monkeypatch.setattr(runner, "check_credentials", lambda: False)
    assert runner.launch({}) is None
    sink.getvalue().encode(encoding)
    assert "SSE content chunks" in sink.getvalue()
    assert "TPOT" not in sink.getvalue()
    if encoding == "cp1252":
        assert sink.getvalue().isascii()
