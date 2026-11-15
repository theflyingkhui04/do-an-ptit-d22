"""Empirical serving gate cannot be bypassed by legacy calibration/fallback."""
import json
import pytest
from flexsds.core import cost_model as cm
from flexsds.core.hardware import HardwareInfo
from flexsds.cli.main import main


def test_synthetic_a10g_rejected_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    path = cm.save_hardware_profile(dict(metadata=dict(provenance='micro_benchmark_calibrated'),
                                        forward_target={'1':16.5}, forward_draft={'1':3.8}), 'sm_86')
    hw = HardwareInfo('Remote A10G','sm_86',22000,22000,'',True)
    assert cm.load_profile_for_hardware(hw, strict=True) == (None, '')
    assert path.exists()


def test_analytical_builtin_not_accepted_as_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    hw = HardwareInfo('Remote L4','sm_89',22000,22000,'',True)
    assert cm.load_profile_for_hardware(hw, strict=True) == (None, '')


@pytest.mark.parametrize('extra', [[], ['--allow-fallback']])
def test_eagle_adaptive_still_requires_empirical_profile(extra, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)
    result = main(['serve', '--backend', 'modal', '--target', 'Qwen/Qwen2.5-7B-Instruct-AWQ',
                   '--draft', 'duckhoi/qwen2.5-7b-eagle3.1-speculator', '--dry-run', *extra])
    assert result == 1
    assert "No measured latency profile" in capsys.readouterr().out


def test_profile_benchmark_cpu_fails_without_success(capsys, monkeypatch):
    monkeypatch.setattr('flexsds.cli.main.detect_hardware', lambda: HardwareInfo('CPU','cpu',0,0,'',False))
    assert main(['profile', '--benchmark']) == 1
    output = capsys.readouterr().out
    assert 'Calibration failed' in output
    assert 'Calibrated & saved' not in output
