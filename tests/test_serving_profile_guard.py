"""Tests for fail-fast guard when serving with uncalibrated hardware profile (CR01 - REQ-HW-03)."""
import pytest
from unittest.mock import patch
from flexsds.core.hardware import HardwareInfo
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
        assert "Chưa có hồ sơ độ trễ cho GPU" in captured.out
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
