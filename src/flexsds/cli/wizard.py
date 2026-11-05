"""Interactive terminal setup wizard for FlexSDS."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from flexsds.core.hardware import detect_hardware, HardwareInfo
    from flexsds.core.compatibility import check_compatibility, CompatibilityReport
    from flexsds.core.cost_model import load_profile_for_hardware
except ModuleNotFoundError:
    from src.flexsds.core.hardware import detect_hardware, HardwareInfo
    from src.flexsds.core.compatibility import check_compatibility, CompatibilityReport
    from src.flexsds.core.cost_model import load_profile_for_hardware


def prompt_choice(question: str, choices: list[str], default_idx: int = 0) -> str:
    """Prompts user to select from a list of choices."""
    print(f"\n\033[1;36m? {question}\033[0m")
    for i, choice in enumerate(choices, 1):
        indicator = "\033[1;32m>\033[0m" if (i - 1) == default_idx else " "
        default_tag = " \033[90m(Default)\033[0m" if (i - 1) == default_idx else ""
        print(f"  {indicator} [{i}] {choice}{default_tag}")
    
    try:
        user_input = input(f"  \033[90mNhập lựa chọn [1-{len(choices)}] (Enter để chọn mặc định): \033[0m").strip()
        if not user_input:
            return choices[default_idx]
        idx = int(user_input) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except Exception:
        pass
    return choices[default_idx]


def prompt_string(question: str, default_val: str) -> str:
    """Prompts user for a string with a default value."""
    print(f"\n\033[1;36m? {question}\033[0m")
    print(f"  \033[90m[Mặc định: {default_val}]\033[0m")
    try:
        user_input = input("  \033[37mNhập giá trị: \033[0m").strip()
        return user_input if user_input else default_val
    except (EOFError, KeyboardInterrupt):
        return default_val


def prompt_bool(question: str, default_val: bool = True) -> bool:
    """Prompts user for a yes/no boolean decision."""
    tag = "[Y/n]" if default_val else "[y/N]"
    print(f"\n\033[1;36m? {question} \033[90m{tag}\033[0m")
    try:
        user_input = input("  \033[37mChọn: \033[0m").strip().lower()
        if not user_input:
            return default_val
        return user_input in ("y", "yes", "true", "1")
    except (EOFError, KeyboardInterrupt):
        return default_val


def run_interactive_wizard(hw_info: HardwareInfo) -> Dict[str, Any]:
    """Guides the user through setting up FlexSDS serving step-by-step."""
    print("\n\033[1;33m--- TRỢ LÝ THIẾT LẬP PHỤC VỤ (FLEXSDS SERVING WIZARD) ---\033[0m")

    # Step 1: Select Action
    action = prompt_choice(
        "Chọn chế độ hoạt động:",
        ["Khởi chạy phục vụ (Launch Serving)", "Chỉ quét phần cứng (Hardware Detection Only)", "Đo đạc hồ sơ GPU (Quick Micro-calibration)"],
        default_idx=0,
    )

    if "Hardware Detection" in action:
        print("\n\033[1;32m✔ Đã hoàn tất quét phần cứng. Thông số hiển thị trên banner.\033[0m")
        return {"action": "detect"}

    if "Micro-calibration" in action:
        print(f"\n\033[1;32m✔ Kích hoạt micro-calibration đo đạc cho {hw_info.device_name} ({hw_info.sm_architecture})...\033[0m")
        return {"action": "profile"}

    # Step 2: Target Model
    target_model = prompt_string(
        "Nhập tên mô hình đích (Target Model):",
        default_val="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )

    # Step 3: Draft Model
    draft_model = prompt_string(
        "Nhập tên mô hình nháp (Draft Model / Speculator Head):",
        default_val="duckhoi/qwen2.5-7b-eagle3.1-speculator",
    )

    # Step 4: Pre-flight Compatibility Guard
    print("\n\033[90mĐang kiểm tra an toàn VRAM & tương thích bảng từ vựng...\033[0m")
    report = check_compatibility(
        target_model=target_model,
        draft_model=draft_model,
        free_vram_mb=hw_info.free_vram_mb,
    )

    print("\n\033[1;37m┌──────────────── BÁO CÁO KIỂM TRA TƯƠNG THÍCH ────────────────┐\033[0m")
    status_color = "\033[32m" if not report.oom_risk else "\033[31m"
    status_text = "AN TOÀN" if not report.oom_risk else "NGUY CƠ OOM"
    print(f"│ Trạng thái VRAM       : {status_color}{status_text}\033[0m (Cần: ~{report.total_vram_needed_mb/1024:.1f}GB / Trống: {report.free_vram_mb/1024:.1f}GB)")
    print(f"│ Phương thức suy đoán  : \033[1;36m{report.draft_method}\033[0m")
    print(f"│ Lệch bảng từ vựng     : {'CÓ (Tự bật --use-heterogeneous-vocab)' if report.use_heterogeneous_vocab else 'ĐỒNG NHẤT'}")
    for warn in report.warnings:
        print(f"│ \033[33mCảnh báo\033[0m              : {warn}")
    for rec in report.recommendations:
        print(f"│ \033[36mKhuyến nghị\033[0m           : {rec}")
    print("\033[1;37m└───────────────────────────────────────────────────────────────┘\033[0m")

    # Step 5: Backend
    backend_choice = prompt_choice(
        "Chọn nền tảng triển khai (Serving Backend):",
        ["Modal Cloud GPU (Serverless L4 - Khuyên dùng)", "Local Dedicated GPU (vLLM cục bộ)"],
        default_idx=0,
    )
    backend = "modal" if "Modal" in backend_choice else "local"

    # Step 6: TreeAdaptive toggle
    enable_adaptive = prompt_bool(
        "Kích hoạt lớp thích ứng phần cứng TreeAdaptive (Hardware-Aware Dynamic k*)?",
        default_val=True,
    )

    # Step 7: Cost profile matching
    cost_model, profile_name = load_profile_for_hardware(hw_info)
    print(f"\n\033[32m✔\033[0m Đã gắn hồ sơ chi phí phần cứng: \033[1m{profile_name}\033[0m")

    # Assemble serving configuration
    config = {
        "flexsds_version": "0.2.0-alpha",
        "action": "serve",
        "backend": backend,
        "target_model": target_model,
        "draft_model": draft_model,
        "draft_method": report.draft_method,
        "use_heterogeneous_vocab": report.use_heterogeneous_vocab,
        "enable_adaptive": enable_adaptive,
        "profile_applied": profile_name,
        "hardware": {
            "device_name": hw_info.device_name,
            "sm_architecture": hw_info.sm_architecture,
            "free_vram_mb": hw_info.free_vram_mb,
        },
    }

    # Save to serving_config.json
    config_file = Path("serving_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n\033[32m✔\033[0m Đã lưu cấu hình phục vụ tại: \033[1m{config_file.resolve()}\033[0m")

    return config
