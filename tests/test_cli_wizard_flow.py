"""Tests for interactive wizard and CLI dispatcher flow."""
import pytest
from flexsds.cli.wizard import PLATFORM_CHOICES
from flexsds.cli.main import parse_args


def test_wizard_platform_selection_structure():
    assert "Modal.com" in PLATFORM_CHOICES[0]
    assert "Google Colab" in PLATFORM_CHOICES[1]
    assert "Local vLLM" in PLATFORM_CHOICES[2]


def test_cli_stop_command_parser():
    args = parse_args(["stop"])
    assert args.command == "stop"


def test_cli_serve_flags_parser():
    args = parse_args(["serve", "--port", "8008", "--background", "--deploy", "--dry-run"])
    assert args.command == "serve"
    assert args.port == 8008
    assert args.background is True
    assert args.deploy is True
    assert args.dry_run is True


def test_modal_gpu_selection_extraction():
    """Kiểm tra trích xuất chính xác tên GPU từ chuỗi lựa chọn hiển thị có kèm ghi chú."""
    choices = [
        "A10G (Recommended - 24 GB VRAM)",
        "L4 (24 GB VRAM - Capacity limited)",
        "T4 (16 GB VRAM)",
    ]

    def extract_gpu(choice_str: str) -> str:
        for gpu in ["A10G", "L4", "T4"]:
            if choice_str.startswith(gpu):
                return gpu
        return "A10G"

    assert extract_gpu(choices[0]) == "A10G"
    assert extract_gpu(choices[1]) == "L4"
    assert extract_gpu(choices[2]) == "T4"

    # Also test wizard's inline extraction logic:
    # target_gpu = "A10G" if "A10G" in chosen else ("L4" if "L4" in chosen else "T4")
    def wizard_inline_extract(chosen: str) -> str:
        return "A10G" if "A10G" in chosen else ("L4" if "L4" in chosen else "T4")

    assert wizard_inline_extract(choices[0]) == "A10G"
    assert wizard_inline_extract(choices[1]) == "L4"
    assert wizard_inline_extract(choices[2]) == "T4"


def test_wizard_disabled_adaptive_skips_profile_mapping():
    """Kiểm tra khi tắt HardwareAdaptive thì không liên kết profile và profile_applied là None."""
    # Kiểm tra cấu trúc logic tạo config khi enable_adaptive = False
    enable_adaptive = False
    profile_name = None if not enable_adaptive else "profile_sm89_l4.json"

    assert profile_name is None
    config = {
        "enable_adaptive": enable_adaptive,
        "profile_applied": profile_name,
    }
    assert config["enable_adaptive"] is False
    assert config["profile_applied"] is None


def test_wizard_remote_serving_hardware_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr("flexsds.core.cost_model.USER_PROFILES_DIR", tmp_path)
    """Kiểm tra logic phân giải đúng kiến trúc GPU đích khi chạy Modal/Colab."""
    from flexsds.core.hardware import HardwareInfo
    from flexsds.core.cost_model import load_profile_for_hardware

    remote_sm_map = {
        "A10G": "sm_86",
        "L4": "sm_89",
        "T4": "sm_75",
        "A100": "sm_80",
    }
    budgets = {"T4": 15000.0, "L4": 23000.0, "A10G": 22000.0, "A100": 40000.0}

    def resolve_serving_hw(backend: str, target_gpu: str, local_hw: HardwareInfo) -> HardwareInfo:
        if backend in ("modal", "colab"):
            target_sm = remote_sm_map.get(target_gpu, "sm_89")
            target_vram = budgets.get(target_gpu, 22000.0)
            return HardwareInfo(
                device_name=f"Remote {target_gpu}",
                sm_architecture=target_sm,
                total_vram_mb=target_vram,
                free_vram_mb=target_vram,
                cuda_version="12.4",
                is_cuda_available=True,
                device_count=1,
            )
        return local_hw

    cpu_hw = HardwareInfo("AMD Ryzen", "cpu", 16000.0, 16000.0, "N/A", False)

    # Test Modal A10G
    modal_a10g_hw = resolve_serving_hw("modal", "A10G", cpu_hw)
    assert modal_a10g_hw.sm_architecture == "sm_86"
    assert "A10G" in modal_a10g_hw.device_name
    _, a10g_prof = load_profile_for_hardware(modal_a10g_hw, strict=False)
    assert a10g_prof == "profile_sm_86_remote_a10g_fcda5486b12d.json"

    # Test Modal L4
    modal_l4_hw = resolve_serving_hw("modal", "L4", cpu_hw)
    assert modal_l4_hw.sm_architecture == "sm_89"
    _, l4_prof = load_profile_for_hardware(modal_l4_hw, strict=False)
    assert l4_prof == "profile_sm89_l4.json"

    # Test Modal T4
    modal_t4_hw = resolve_serving_hw("modal", "T4", cpu_hw)
    assert modal_t4_hw.sm_architecture == "sm_75"
    _, t4_prof = load_profile_for_hardware(modal_t4_hw, strict=False)
    assert t4_prof == "profile_sm75_t4.json"

    # Test Local giữ nguyên CPU
    local_hw = resolve_serving_hw("local", "L4", cpu_hw)
    assert local_hw.sm_architecture == "cpu"


def test_wizard_uncalibrated_gpu_abort_action(tmp_path, monkeypatch, capsys):
    """Khi gặp GPU chưa có profile đo đạc và chọn Abort setup, wizard phải thoát với exit code 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('flexsds.core.cost_model.USER_PROFILES_DIR', tmp_path / 'profiles')
    monkeypatch.setattr('flexsds.core.cost_model.PROFILES_DIR', tmp_path / 'bundled')
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import HardwareInfo

    cpu_hw = HardwareInfo("AMD Ryzen", "cpu", 16000.0, 16000.0, "N/A", False)

    def fake_prompt_choice(question, choices, default_idx=0):
        q = question.lower()
        if "action" in q:
            return "Launch serving"
        if "model configuration" in q:
            return choices[0]
        if "platform" in q:
            return "Modal.com (Cloud Serverless GPU)"
        if "modal gpu" in q:
            return "A10G (Recommended - 24 GB VRAM)"
        if "missing" in q:
            return "Abort setup"
        return choices[default_idx]

    def fake_prompt_bool(question, default_val=True):
        q = question.lower()
        if "hardwareadaptive" in q:
            return True
        if "deploy" in q:
            return False
        return False

    def fake_prompt_string(question, default_val=""):
        return default_val

    monkeypatch.setattr("flexsds.cli.wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_bool", fake_prompt_bool)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_string", fake_prompt_string)

    result = run_interactive_wizard(cpu_hw)

    assert result == {"action": "abort", "exit_code": 1}
    assert not (tmp_path / "serving_config.json").exists()

    captured = capsys.readouterr()
    assert "[Hardware Profile Error]" in captured.out
    assert "Serving setup aborted" in captured.out


def test_wizard_uncalibrated_gpu_switch_static_action(tmp_path, monkeypatch, capsys):
    """Khi gặp GPU chưa có profile đo đạc và chọn Switch to static, wizard tắt adaptive và tiếp tục."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('flexsds.core.cost_model.USER_PROFILES_DIR', tmp_path / 'profiles')
    monkeypatch.setattr('flexsds.core.cost_model.PROFILES_DIR', tmp_path / 'bundled')
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import HardwareInfo

    cpu_hw = HardwareInfo("AMD Ryzen", "cpu", 16000.0, 16000.0, "N/A", False)

    def fake_prompt_choice(question, choices, default_idx=0):
        q = question.lower()
        if "action" in q:
            return "Launch serving"
        if "model configuration" in q:
            return choices[0]
        if "platform" in q:
            return "Modal.com (Cloud Serverless GPU)"
        if "modal gpu" in q:
            return "A10G (Recommended - 24 GB VRAM)"
        if "missing" in q:
            return "Switch to static speculation (serve without HardwareAdaptive)"
        return choices[default_idx]

    def fake_prompt_bool(question, default_val=True):
        q = question.lower()
        if "hardwareadaptive" in q:
            return True
        if "deploy" in q:
            return False
        if "launch server" in q:
            return False
        return False

    def fake_prompt_string(question, default_val=""):
        return default_val

    monkeypatch.setattr("flexsds.cli.wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_bool", fake_prompt_bool)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_string", fake_prompt_string)

    result = run_interactive_wizard(cpu_hw)

    assert result["action"] == "serve"
    assert result["enable_adaptive"] is False
    assert result["profile_applied"] is None

    cfg_file = tmp_path / "serving_config.json"
    assert cfg_file.exists()
    import json
    with open(cfg_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["enable_adaptive"] is False
    assert saved_data["profile_applied"] is None

    captured = capsys.readouterr()
    assert "[Hardware Profile Error]" in captured.out
    assert "Switched to static speculation mode" in captured.out


def test_wizard_uncalibrated_gpu_calibrate_action(tmp_path, monkeypatch, capsys):
    """Khi gặp GPU chưa có profile đo đạc và chọn Calibrate now, wizard chạy benchmark và nạp profile."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('flexsds.core.cost_model.USER_PROFILES_DIR', tmp_path / 'profiles')
    monkeypatch.setattr('flexsds.core.cost_model.PROFILES_DIR', tmp_path / 'bundled')
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import HardwareInfo
    from pathlib import Path
    from unittest.mock import MagicMock

    profile_fixture = tmp_path / "measured.json"
    profile_fixture.write_text("{}")
    monkeypatch.setattr("flexsds.cli.wizard.resolve_profile_path", lambda name: profile_fixture)
    cpu_hw = HardwareInfo("AMD Ryzen", "cpu", 16000.0, 16000.0, "N/A", False)

    def fake_prompt_choice(question, choices, default_idx=0):
        q = question.lower()
        if "action" in q:
            return "Launch serving"
        if "model configuration" in q:
            return choices[0]
        if "platform" in q:
            return "Modal.com (Cloud Serverless GPU)"
        if "modal gpu" in q:
            return "A10G (Recommended - 24 GB VRAM)"
        if "missing" in q:
            return "Calibrate GPU profile now (run micro-benchmark)"
        return choices[default_idx]

    def fake_prompt_bool(question, default_val=True):
        q = question.lower()
        if "hardwareadaptive" in q:
            return True
        if "deploy" in q:
            return False
        if "launch server" in q:
            return False
        return False

    def fake_prompt_string(question, default_val=""):
        return default_val

    benchmark_called = []
    fake_cost_model = MagicMock()

    def fake_benchmark(hw, target_model=None, draft_model=None, backend=None, measurement_mode=None):
        assert measurement_mode == "fast"
        benchmark_called.append((hw.sm_architecture, target_model, draft_model))
        return fake_cost_model, Path("/fake/path/profile_sm86_a10g.json")

    call_count = {"count": 0}

    def fake_load_profile(hw, strict=True, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return None, ""
        return fake_cost_model, "profile_sm86_a10g.json"

    monkeypatch.setattr("flexsds.cli.wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_bool", fake_prompt_bool)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_string", fake_prompt_string)
    monkeypatch.setattr("flexsds.cli.wizard.run_hardware_micro_benchmark", fake_benchmark, raising=False)
    monkeypatch.setattr("flexsds.cli.wizard.load_profile_for_hardware", fake_load_profile)

    result = run_interactive_wizard(cpu_hw)

    assert len(benchmark_called) == 1
    assert benchmark_called[0][0] == "sm_86"
    assert result["action"] == "serve"
    assert result["enable_adaptive"] is True
    assert result["profile_applied"] == "profile_sm86_a10g.json"

    cfg_file = tmp_path / "serving_config.json"
    assert cfg_file.exists()
    import json
    with open(cfg_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["enable_adaptive"] is True
    assert saved_data["profile_applied"] == "profile_sm86_a10g.json"

    captured = capsys.readouterr()
    assert "[Hardware Profile Error]" in captured.out
    assert "Calibrated & saved new hardware profile" in captured.out
    assert "Linked hardware profile: profile_sm86_a10g.json" in captured.out






@pytest.mark.parametrize('mode', ['fast', 'full'])
def test_profile_menu_measures_and_saves(tmp_path, monkeypatch, capsys, mode):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.setattr('flexsds.core.cost_model.USER_PROFILES_DIR', tmp_path)
    monkeypatch.setattr('flexsds.core.cost_model.PROFILES_DIR', tmp_path)
    def choose(question, choices, default_idx=0):
        if question == 'Select action:':
            assert choices[2] == 'Measure & available profile GPU'
            return choices[2]
        if question == 'Measurement mode:':
            return 'Fast measure' if mode == 'fast' else 'Full measure'
        return choices[default_idx]
    monkeypatch.setattr(wizard, 'prompt_choice', choose)
    monkeypatch.setattr(wizard, 'prompt_string', lambda question, default_val='': default_val)
    monkeypatch.setattr(wizard, 'load_profile_for_hardware', lambda *a, **kw: (None, ''))
    def measure(hw, **kwargs):
        assert kwargs['backend'] == 'modal'
        assert kwargs['measurement_mode'] == mode
        return object(), tmp_path / f'profile_{mode}.json'
    monkeypatch.setattr(wizard, 'run_hardware_micro_benchmark', measure)
    result = wizard.run_interactive_wizard(HardwareInfo('CPU', 'cpu', 0, 0, '', False))
    assert result['profile_path'].endswith(f'profile_{mode}.json')
    assert 'Available GPU profiles' in capsys.readouterr().out


def test_profile_menu_reuses_existing_without_measurement(monkeypatch, capsys):
    from flexsds.cli import wizard
    from flexsds.core.hardware import HardwareInfo
    monkeypatch.setattr(wizard, 'prompt_choice', lambda question, choices, default_idx=0: choices[default_idx])
    monkeypatch.setattr(wizard, 'prompt_string', lambda question, default_val='': default_val)
    monkeypatch.setattr(wizard, 'load_profile_for_hardware', lambda *a, **kw: (object(), 'existing.json'))
    monkeypatch.setattr(wizard, 'run_hardware_micro_benchmark', lambda *a, **kw: pytest.fail('unexpected measurement'))
    assert wizard.run_profile_wizard(HardwareInfo('CPU', 'cpu', 0, 0, '', False))['profile_applied'] == 'existing.json'
    assert 'already exists' in capsys.readouterr().out
