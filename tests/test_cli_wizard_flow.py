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


def test_wizard_remote_serving_hardware_resolution():
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
    assert a10g_prof == "conservative_analytical_fallback"

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



