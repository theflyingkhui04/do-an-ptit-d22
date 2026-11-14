"""Tests for fail-fast guard when serving with uncalibrated hardware profile (CR01 - REQ-HW-03)."""
import pytest
from unittest.mock import patch
from flexsds.core.hardware import HardwareInfo
from flexsds.core.cost_model import run_hardware_micro_benchmark, load_profile_for_hardware
from flexsds.cli.main import main, get_serving_cost_profile


def test_get_serving_cost_profile_strict_and_fallback():
    unknown_hw = HardwareInfo(
        device_name="NVIDIA Custom GPU",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    # 1. Chế độ nghiêm ngặt (strict = True khi allow_fallback=False): trả về (None, "")
    cost_model, prof_name = get_serving_cost_profile(unknown_hw, allow_fallback=False)
    assert cost_model is None
    assert prof_name == ""

    # 2. Chế độ cho phép fallback (allow_fallback=True): trả về mô hình dè dặt
    cost_model, prof_name = get_serving_cost_profile(unknown_hw, allow_fallback=True)
    assert cost_model is not None
    assert prof_name == "conservative_analytical_fallback"


def test_get_serving_cost_profile_uncalibrated_gpu_fails_in_strict_mode():
    """Kiểm tra GPU chưa có hồ sơ thực nghiệm (như A10G sm_86) trả về (None, '') khi strict=True."""
    a10g_hw = HardwareInfo(
        device_name="Remote A10G",
        sm_architecture="sm_86",
        total_vram_mb=22000.0,
        free_vram_mb=22000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    cost_model, prof_name = get_serving_cost_profile(a10g_hw, allow_fallback=False)
    assert cost_model is None
    assert prof_name == ""


def test_serve_fails_fast_when_missing_profile_without_fallback_flag(capsys):
    unknown_hw = HardwareInfo(
        device_name="NVIDIA Custom GPU",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    with patch("flexsds.cli.main.detect_hardware", return_value=unknown_hw):
        # Chạy lệnh serve không truyền --allow-fallback
        ret = main(["serve", "--backend", "local", "--dry-run"])
        assert ret == 1
        captured = capsys.readouterr()
        assert (
            "No measured latency profile for GPU" in captured.out
            or "No latency profile for GPU" in captured.out
            or "Chưa có hồ sơ độ trễ cho GPU" in captured.out
        )
        assert "flexsds profile --benchmark" in captured.out


def test_serve_proceeds_when_allow_fallback_flag_provided(capsys):
    unknown_hw = HardwareInfo(
        device_name="NVIDIA Custom GPU",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    with patch("flexsds.cli.main.detect_hardware", return_value=unknown_hw):
        # Chạy lệnh serve có truyền cờ --allow-fallback
        ret = main([
            "serve",
            "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "--backend", "local",
            "--dry-run",
            "--allow-fallback",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Conservative Fallback" in captured.out or "conservative_analytical_fallback" in captured.out


def test_serve_fails_fast_when_draft_is_incompatible_architecture(capsys):
    # Khóa cứng chặn phục vụ khi mô hình nháp không thuộc kiến trúc EAGLE (AC-05)
    with patch("flexsds.core.compatibility.get_model_config", side_effect=[
        {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064},
        {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064},
    ]):
        ret = main([
            "serve",
            "--backend", "local",
            "--target", "Qwen/Qwen2.5-7B-Instruct",
            "--draft", "Qwen/Qwen2.5-0.5B-Instruct",
            "--allow-fallback",
        ])
        assert ret == 2
        captured = capsys.readouterr()
        assert "Launch blocked" in captured.out or "incompatible" in captured.out


def test_serve_modal_fails_fast_when_gpu_profile_missing(capsys):
    ret = main(["serve", "--backend", "modal", "--gpu", "A10G", "--dry-run"])
    assert ret == 1
    captured = capsys.readouterr()
    assert (
        "No measured latency profile for GPU" in captured.out
        or "No latency profile for GPU" in captured.out
    )
    assert "A10G" in captured.out
    assert "sm_86" in captured.out
    assert "flexsds profile --benchmark" in captured.out


def test_serve_modal_proceeds_with_no_adaptive(capsys):
    ret = main(["serve", "--backend", "modal", "--gpu", "A10G", "--no-adaptive", "--dry-run"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Preview only; not a readiness check" in captured.out


def test_micro_benchmark_calibrates_sm86_a10g(tmp_path, monkeypatch):
    """Kiểm tra quá trình micro-benchmark hiệu chuẩn hồ sơ phần cứng cho sm_86 (A10G)."""
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    a10g_hw = HardwareInfo(
        device_name="Remote A10G",
        sm_architecture="sm_86",
        total_vram_mb=22000.0,
        free_vram_mb=22000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    cost_model_bench, saved_path = run_hardware_micro_benchmark(a10g_hw)
    try:
        assert saved_path.exists()
        cost_model, profile_name = load_profile_for_hardware(a10g_hw, strict=True)
        assert cost_model is not None
        assert "sm86" in profile_name or "sm_86" in profile_name
        c_t, c_d = cost_model.get_forward_costs(batch_size=1)
        assert c_t == 16.5
        assert c_d == 3.8
    finally:
        if saved_path.exists():
            saved_path.unlink()


