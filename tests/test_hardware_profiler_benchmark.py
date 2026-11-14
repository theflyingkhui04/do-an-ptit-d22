import pytest
from pathlib import Path
from flexsds.core.hardware import HardwareInfo
from flexsds.core.cost_model import (
    run_hardware_micro_benchmark,
    FlexCostModel,
    USER_PROFILES_DIR,
)
from flexsds.cli.main import main


def test_run_hardware_micro_benchmark_creates_profile(tmp_path, monkeypatch):
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    mock_hw = HardwareInfo(
        device_name="NVIDIA RTX 4090",
        sm_architecture="sm_89",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    model, saved_path = run_hardware_micro_benchmark(
        hw_info=mock_hw,
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        batch_sizes=[1, 2, 4],
        target_dir=tmp_path,
    )
    assert isinstance(model, FlexCostModel)
    assert saved_path.exists()
    assert "sm_89" in saved_path.name
    assert model.forward_target[1] > 0
    assert model.forward_draft[1] > 0


def test_cli_profile_with_benchmark_flag(capsys, monkeypatch, tmp_path):
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    ret = main(["profile", "--benchmark"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Calibrated & saved new hardware profile" in captured.out


def test_run_hardware_micro_benchmark_default_batch_sizes(tmp_path, monkeypatch):
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    mock_hw = HardwareInfo(
        device_name="NVIDIA A100-SXM4-40GB",
        sm_architecture="sm_80",
        total_vram_mb=40960.0,
        free_vram_mb=38000.0,
        cuda_version="12.2",
        is_cuda_available=True,
    )
    model, saved_path = run_hardware_micro_benchmark(
        hw_info=mock_hw,
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert isinstance(model, FlexCostModel)
    assert saved_path.exists()
    assert saved_path.parent == tmp_path
    assert 8 in model.forward_target
    assert 8 in model.forward_draft
    assert model.forward_target[8] > model.forward_target[1]


def test_core_init_exports_run_hardware_micro_benchmark():
    import flexsds.core as core
    assert hasattr(core, "run_hardware_micro_benchmark")
    assert callable(core.run_hardware_micro_benchmark)
