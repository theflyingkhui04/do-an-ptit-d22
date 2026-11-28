"""Empirical serving gate cannot be bypassed by legacy calibration/fallback."""
import json
import pytest
from flexsds.core import cost_model as cm
from flexsds.core.hardware import HardwareInfo
from flexsds.cli.main import main


def test_synthetic_a10_rejected_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    monkeypatch.setattr(cm, 'PROFILES_DIR', tmp_path / 'bundled')
    path = cm.save_hardware_profile(dict(metadata=dict(provenance='micro_benchmark_calibrated'),
                                        forward_target={'1':16.5}, forward_draft={'1':3.8}), 'sm_86')
    hw = HardwareInfo('Remote A10','sm_86',22000,22000,'',True)
    assert cm.load_profile_for_hardware(hw, strict=True) == (None, '')
    assert path.exists()


def test_analytical_builtin_not_accepted_as_measured(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(cm, 'USER_PROFILES_DIR', tmp_path)
    analytical = tmp_path / 'analytical-only'
    analytical.mkdir()
    for name in ('profile_sm89_l4.json', 'profile_sm75_t4.json'):  # vendor estimates, no measured profile
        shutil.copy(cm.PROFILES_DIR / name, analytical / name)
    monkeypatch.setattr(cm, 'PROFILES_DIR', analytical)
    hw = HardwareInfo('Remote L40S','sm_89',46000,46000,'',True)
    assert cm.load_profile_for_hardware(hw, strict=True) == (None, '')


@pytest.mark.parametrize('extra', [[], ['--allow-fallback']])
def test_eagle_adaptive_still_requires_empirical_profile(extra, capsys, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import flexsds.cli.main as cli_main
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)
    monkeypatch.setattr(cm, "PROFILES_DIR", tmp_path / "bundled")
    fake_report = SimpleNamespace(
        draft_method="eagle3", eagle_version="EAGLE-3.1",
        target_revision="a" * 40, draft_revision="b" * 40,
        use_heterogeneous_vocab=False, identified_method="eagle3_1",
        oom_risk=False, warnings=[], recommendations=[],
        summary=lambda: "Compatibility: COMPATIBLE",
    )
    fake_preflight = SimpleNamespace(report=fake_report, launch_reasons=lambda allow_oom: [])
    monkeypatch.setattr(cli_main, "evaluate_serving_preflight", lambda *a, **kw: fake_preflight)
    result = main(['serve', '--backend', 'modal', '--gpu', 'T4', '--target', 'Qwen/Qwen2.5-7B-Instruct-AWQ',
                   '--draft', 'duckhoi/qwen2.5-7b-eagle3.1-speculator', '--dry-run', *extra])
    assert result == 1
    assert "No measured latency profile" in capsys.readouterr().out


def test_profile_benchmark_cpu_fails_without_success(capsys, monkeypatch):
    monkeypatch.setattr('flexsds.cli.main.detect_hardware', lambda: HardwareInfo('CPU','cpu',0,0,'',False))
    assert main(['profile', '--benchmark']) == 1
    output = capsys.readouterr().out
    assert 'Calibration failed' in output
    assert 'Calibrated & saved' not in output
