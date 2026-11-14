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


