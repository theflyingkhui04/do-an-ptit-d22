"""Trợ lý tương tác thiết lập máy chủ FlexSDS (Interactive Wizard)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flexsds.core.hardware import detect_hardware, HardwareInfo
    from flexsds.core.compatibility import (
        check_compatibility,
        pre_download_guard,
        CompatibilityReport,
    )
    from flexsds.core.cost_model import load_profile_for_hardware
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.colab_runner import ColabServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds.core.hardware import detect_hardware, HardwareInfo
    from src.flexsds.core.compatibility import (
        check_compatibility,
        pre_download_guard,
        CompatibilityReport,
    )
    from src.flexsds.core.cost_model import load_profile_for_hardware
    from src.flexsds.server.local_runner import LocalServingRunner
    from src.flexsds.server.colab_runner import ColabServingRunner
    from src.flexsds.server.modal_runner import ModalServingRunner

PLATFORM_CHOICES: List[str] = [
    "Modal.com (Serverless GPU)",
    "Google Colab (Miễn phí qua notebook)",
    "Tự dùng GPU của thiết bị này",
]


def prompt_choice(question: str, choices: list[str], default_idx: int = 0) -> str:
    """Yêu cầu người dùng chọn một mục trong danh sách."""
    print(f"\n\033[1;36m? {question}\033[0m")
    for i, choice in enumerate(choices, 1):
        indicator = "\033[1;32m>\033[0m" if (i - 1) == default_idx else " "
        default_tag = " \033[90m(Mặc định)\033[0m" if (i - 1) == default_idx else ""
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
    """Yêu cầu người dùng nhập một chuỗi ký tự."""
    print(f"\n\033[1;36m? {question}\033[0m")
    print(f"  \033[90m[Mặc định: {default_val}]\033[0m")
    try:
        user_input = input("  \033[37mNhập giá trị: \033[0m").strip()
        return user_input if user_input else default_val
    except (EOFError, KeyboardInterrupt):
        return default_val


def prompt_bool(question: str, default_val: bool = True) -> bool:
    """Yêu cầu người dùng chọn Yes/No (Đồng ý/Từ chối)."""
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
    """Hướng dẫn người dùng thiết lập và khởi chạy phục vụ từng bước."""
    print("\n\033[1;33m--- Trợ lý thiết lập phục vụ (FlexSDS serving wizard) ---\033[0m")

    # Bước 1: Chọn chế độ hoạt động
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

    # Bước 2: Mô hình đích
    target_model = prompt_string(
        "Nhập tên mô hình đích (hoặc đường dẫn Hugging Face):",
        default_val="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )

    # Bước 3: Đầu nháp EAGLE
    draft_model = prompt_string(
        "Nhập tên đầu nháp EAGLE (Speculator Head):",
        default_val="duckhoi/qwen2.5-7b-eagle3.1-speculator",
    )

    # Bước 4: Kiểm tra an toàn trước khi tải (Pre-download Guard) & Báo cáo tương thích
    print("\n\033[90mĐang kiểm tra an toàn VRAM & tương thích bảng từ vựng...\033[0m")
    pre_guard = pre_download_guard(
        target_model=target_model,
        draft_model=draft_model,
        free_vram_mb=hw_info.free_vram_mb,
    )

    report = check_compatibility(
        target_model=target_model,
        draft_model=draft_model,
        free_vram_mb=hw_info.free_vram_mb,
    )

    if not pre_guard.is_safe and pre_guard.warning:
        print(f"\n\033[1;31m[Cảnh báo VRAM]\033[0m {pre_guard.warning}")
        if pre_guard.suggested_model:
            print(f"\033[1;33m[Khuyến nghị]\033[0m Bạn nên sử dụng phiên bản lượng tử hóa: \033[1m{pre_guard.suggested_model}\033[0m")
            use_suggested = prompt_bool(f"Chuyển sang dùng mô hình {pre_guard.suggested_model}?", default_val=True)
            if use_suggested:
                target_model = pre_guard.suggested_model
                report = check_compatibility(target_model, draft_model, free_vram_mb=hw_info.free_vram_mb)

    print("\n\033[1;37m┌──────────────── Báo cáo kiểm tra tương thích ────────────────┐\033[0m")
    status_color = "\033[32m" if not report.oom_risk else "\033[31m"
    status_text = "AN TOÀN" if not report.oom_risk else "NGUY CƠ OOM"
    print(f"│ Trạng thái VRAM       : {status_color}{status_text}\033[0m (Cần: ~{report.total_vram_needed_mb/1024:.1f} GB / Trống: {report.free_vram_mb/1024:.1f} GB)")
    print(f"│ Phương thức suy đoán  : \033[1;36m{report.draft_method}\033[0m")
    tokenizer_status = "CÓ (Đồng nhất)" if not report.use_heterogeneous_vocab else "KHÔNG (Tự động bật --use-heterogeneous-vocab)"
    print(f"│ Dùng chung tokenizer  : \033[1;32m{tokenizer_status}\033[0m")
    for warn in report.warnings:
        print(f"│ \033[33mCảnh báo\033[0m              : {warn}")
        for rec in report.recommendations:
            print(f"│ \033[36mKhuyến nghị\033[0m           : {rec}")
    print("\033[1;37m└───────────────────────────────────────────────────────────────┘\033[0m")

    # Bước 5: Chọn nền tảng triển khai
    platform_choice = prompt_choice(
        "Chọn nền tảng triển khai (Serving Platform):",
        PLATFORM_CHOICES,
        default_idx=0,
    )

    backend = "modal"
    target_gpu = "L4"
    colab_tunnel_url = ""
    modal_deploy = False

    if "Modal.com" in platform_choice:
        backend = "modal"
        gpu_choice = prompt_choice("Chọn loại GPU trên Modal:", ["L4 (Khuyên dùng)", "A10G", "T4"], default_idx=0)
        target_gpu = gpu_choice.split()[0]

        modal_runner = ModalServingRunner()
        if not modal_runner.check_credentials():
            print("\n\033[33m[Lưu ý BYOC Modal]\033[0m Chưa tìm thấy thông tin xác thực Modal.")
            token_id = prompt_string("Nhập MODAL_TOKEN_ID (hoặc Enter để cấu hình sau):", "")
            if token_id:
                token_secret = prompt_string("Nhập MODAL_TOKEN_SECRET:", "")
                os.environ["MODAL_TOKEN_ID"] = token_id
                os.environ["MODAL_TOKEN_SECRET"] = token_secret

        mode_choice = prompt_choice(
            "Chọn chế độ chạy trên Modal:",
            ["Tạm thời (Ephemeral `modal run` - Dành cho phát triển/thử nghiệm)", "Cố định (Persistent `modal deploy` - Phục vụ sản xuất)"],
            default_idx=0,
        )
        modal_deploy = "Cố định" in mode_choice

    elif "Google Colab" in platform_choice:
        backend = "colab"
        gpu_choice = prompt_choice("Chọn loại GPU trên Google Colab:", ["T4 GPU (Miễn phí)", "A100 / L4 GPU (Colab Pro)"], default_idx=0)
        target_gpu = "T4" if "T4" in gpu_choice else "L4"

        colab_runner = ColabServingRunner()
        instructions = colab_runner.get_colab_instructions(target_gpu=target_gpu)
        print("\n\033[1;33m--- Hướng dẫn khởi chạy Google Colab ---\033[0m")
        for step in instructions["steps"]:
            print(f"  {step}")
        print(f"\n  Đường dẫn notebook: \033[1;34m{instructions['notebook_url']}\033[0m\n")

        colab_tunnel_url = prompt_string("Dán URL Cloudflare tunnel sau khi notebook chạy xong (Enter để bỏ qua):", "")
        if colab_tunnel_url:
            print("Đang kiểm tra kết nối tới đường hầm...")
            if colab_runner.verify_tunnel_health(colab_tunnel_url):
                print("\033[32m✔ Đường hầm Cloudflare hoạt động tốt!\033[0m")
            else:
                print("\033[33m⚠ Chưa nhận được phản hồi từ endpoint (có thể máy chủ đang tải weights).\033[0m")

    else:
        backend = "local"
        if not hw_info.is_cuda_available:
            print("\n\033[1;31m[Lỗi phần cứng]\033[0m Không phát hiện card đồ họa NVIDIA CUDA khả dụng trên thiết bị này.")
            print("FlexSDS khuyến nghị chuyển sang nền tảng [1] Modal.com hoặc [2] Google Colab.")
        else:
            print(f"\n\033[32m✔ Đã nhận diện GPU cục bộ: {hw_info.device_name} ({hw_info.free_vram_mb/1024:.1f} GB VRAM khả dụng)\033[0m")

    # Bước 6: Bật TreeAdaptive
    enable_adaptive = prompt_bool(
        "Kích hoạt lớp thích ứng phần cứng TreeAdaptive (Hardware-Aware Dynamic k*)?",
        default_val=True,
    )

    # Bước 7: Ánh xạ hồ sơ chi phí phần cứng
    cost_model, profile_name = load_profile_for_hardware(hw_info)
    print(f"\n\033[32m✔\033[0m Đã gắn hồ sơ chi phí phần cứng: \033[1m{profile_name}\033[0m")

    # Tạo tệp cấu hình serving_config.json
    config: Dict[str, Any] = {
        "flexsds_version": "0.2.0-alpha",
        "action": "serve",
        "backend": backend,
        "target_model": target_model,
        "draft_model": draft_model,
        "draft_method": report.draft_method,
        "use_heterogeneous_vocab": report.use_heterogeneous_vocab,
        "enable_adaptive": enable_adaptive,
        "profile_applied": profile_name,
        "target_gpu": target_gpu,
        "colab_tunnel_url": colab_tunnel_url,
        "modal_deploy": modal_deploy,
        "hardware": {
            "device_name": hw_info.device_name,
            "sm_architecture": hw_info.sm_architecture,
            "free_vram_mb": hw_info.free_vram_mb,
        },
    }

    config_file = Path("serving_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n\033[32m✔\033[0m Đã lưu cấu hình phục vụ tại: \033[1m{config_file.resolve()}\033[0m")

    # Bước 8: Hỏi xác nhận khởi chạy máy chủ ngay lập tức
    should_launch = prompt_bool("Khởi chạy máy chủ ngay bây giờ?", default_val=True)
    if should_launch:
        if backend == "local":
            local_runner = LocalServingRunner()
            run_in_bg = prompt_bool("Chạy máy chủ ở chế độ nền (Background Daemon)?", default_val=False)
            local_runner.launch(config, background=run_in_bg)
        elif backend == "modal":
            modal_runner = ModalServingRunner()
            modal_runner.launch(config, deploy=modal_deploy)
        elif backend == "colab":
            print("\n\033[32m✔ Bạn có thể gửi yêu cầu suy luận trực tiếp tới URL Colab đã cấu hình.\033[0m")

    return config
