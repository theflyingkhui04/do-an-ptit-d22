"""Tests for fail-fast guard when serving with uncalibrated hardware profile (CR01 - REQ-HW-03)."""
import pytest
from unittest.mock import patch
from flexsds.core.hardware import HardwareInfo
from flexsds.cli.main import main


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
        ret = main(["serve", "--backend", "local", "--dry-run", "--allow-fallback"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Conservative Fallback" in captured.out or "conservative_analytical_fallback" in captured.out
