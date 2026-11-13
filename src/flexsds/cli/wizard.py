"""Interactive serving wizard for FlexSDS with arrow-key navigation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

try:
    from flexsds.cli.output import print_output as print
    from flexsds.core.hardware import detect_hardware, HardwareInfo
    from flexsds.core.compatibility import (
        evaluate_serving_preflight,
        ServingBlockedError,
    )
    from flexsds.core.cost_model import load_profile_for_hardware
    from flexsds.server.local_runner import LocalServingRunner, finish_launch
    from flexsds.server.colab_runner import ColabServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
    from src.flexsds.core.hardware import detect_hardware, HardwareInfo
    from src.flexsds.core.compatibility import (
        evaluate_serving_preflight,
        ServingBlockedError,
    )
    from src.flexsds.core.cost_model import load_profile_for_hardware
    from src.flexsds.server.local_runner import LocalServingRunner, finish_launch
    from src.flexsds.server.colab_runner import ColabServingRunner
    from src.flexsds.server.modal_runner import ModalServingRunner


PLATFORM_CHOICES = [
    "Modal.com (Serverless GPU đám mây)",
    "Google Colab (GPU miễn phí T4/L4)",
    "Tự dùng GPU nội bộ (Local vLLM)",
]


def strip_ansi(s: str) -> str:
    """Removes ANSI escape codes to calculate visible string length."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def render_box(title: str, items: list[tuple[str, str, str]], min_width: int = 56) -> str:
    """Renders a self-adjusting border box with perfectly aligned borders.

    items: list of (label, value, value_ansi_color)
    """
    max_label_w = max(len(strip_ansi(label)) for label, _, _ in items) if items else 12
    label_w = max(max_label_w, 14)

    max_content_w = len(title) + 6
    for label, val, _ in items:
        line_w = label_w + len(strip_ansi(val)) + 5
        if line_w > max_content_w:
            max_content_w = line_w

    inner_w = max(max_content_w, min_width)

    title_strip = strip_ansi(title)
    top_left = "─── "
    remaining = inner_w - len(top_left) - len(title_strip) - 1
    top_right = "─" * max(2, remaining)
    actual_inner = len(top_left) + len(title_strip) + 1 + len(top_right)

    top = f"\033[90m┌{top_left}\033[1;37m{title_strip}\033[0m\033[90m {top_right}┐\033[0m"
    bottom = "\033[90m└" + "─" * actual_inner + "┘\033[0m"

    lines = [top]
    for label, val, color in items:
        clean_label = strip_ansi(label)
        clean_val = strip_ansi(val)
        prefix = f" {clean_label:<{label_w}} : "
        used = len(prefix) + len(clean_val)
        pad = " " * max(1, actual_inner - used)
        row = f"\033[90m│\033[0m{prefix}{color}{clean_val}\033[0m{pad}\033[90m│\033[0m"
        lines.append(row)

    lines.append(bottom)
    return "\n".join(lines)


def prompt_choice(question: str, choices: list[str], default_idx: int = 0) -> str:
    """Prompt user to select an option using arrow keys and Enter."""
    if not sys.stdin.isatty() or not HAS_TERMIOS:
        print(f"\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{choices[default_idx]}\033[0m")
        return choices[default_idx]

    current = default_idx
    n = len(choices)

    print(f"\n\033[1;37m? {question}\033[0m\n", end="")
    print("\033[?25l", end="")  # Hide cursor
    sys.stdout.flush()

    def render(first: bool = False) -> None:
        if not first:
            print(f"\033[{n}A", end="")
        for i, choice in enumerate(choices):
            print("\r\033[K", end="")
            if i == current:
                print(f"  \033[1;37m> {choice}\033[0m\n", end="")
            else:
                print(f"    \033[90m{choice}\033[0m\n", end="")
        sys.stdout.flush()

    render(first=True)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":  # Up arrow
                        current = (current - 1) % n
                        render()
                    elif ch3 == "B":  # Down arrow
                        current = (current + 1) % n
                        render()
            elif ch in ("k", "K"):
                current = (current - 1) % n
                render()
            elif ch in ("j", "J"):
                current = (current + 1) % n
                render()
            elif ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < n:
                    current = idx
                    render()
                    break
            elif ch in ("\r", "\n", " "):
                break
            elif ch == "\x03":  # Ctrl+C
                print("\033[?25h\n", end="")
                sys.exit(0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\033[?25h", end="")  # Show cursor
        sys.stdout.flush()

    # Collapse into clean single confirmed line
    print(f"\033[{n+1}A\r\033[K\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{choices[current]}\033[0m\n", end="")
    for _ in range(n):
        print("\033[K\n", end="")
    print(f"\033[{n}A", end="")
    sys.stdout.flush()

    return choices[current]


def prompt_string(question: str, default_val: str = "") -> str:
    """Prompt for text input with a default fallback."""
    if not sys.stdin.isatty():
        print(f"\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{default_val}\033[0m")
        return default_val

    hint = f" \033[90m[{default_val}]\033[0m" if default_val else ""
    try:
        print(f"\n\033[1;37m? {question}\033[0m{hint}: ", end="")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
        val = user_input if user_input else default_val
        print(f"\033[1A\r\033[K\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{val}\033[0m\n", end="")
        sys.stdout.flush()
        return val
    except (EOFError, KeyboardInterrupt):
        print("\n", end="")
        sys.exit(0)


def prompt_bool(question: str, default_val: bool = True) -> bool:
    """Prompt for Yes/No using arrow-key selection."""
    choices = ["Yes", "No"] if default_val else ["No", "Yes"]
    selected = prompt_choice(question, choices, default_idx=0)
    return selected == "Yes"


def run_interactive_wizard(hw_info: HardwareInfo) -> Dict[str, Any]:
    """Guides the user through setting up and launching FlexSDS serving."""
    print("\n\033[1;37m--- FlexSDS Serving Wizard ---\033[0m")

    # Step 1: Select Action
    action = prompt_choice(
        "Select action:",
        ["Launch serving", "Detect hardware", "Benchmark & profile GPU"],
        default_idx=0,
    )

    if action == "Detect hardware":
        print(f"\n\033[32m✔ Hardware specs: {hw_info.summary()}\033[0m")
        return {"action": "detect"}

    if action == "Benchmark & profile GPU":
        cost_model, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[32m✔ Hardware profile loaded: {profile_name}\033[0m")
        return {"action": "profile"}

    # Step 2: Target & Draft Model Configuration
    model_mode = prompt_choice(
        "Model configuration:",
        [
            "Default: Qwen2.5-7B-AWQ + EAGLE-3.1",
            "Custom Hugging Face model",
        ],
        default_idx=0,
    )

    if "Default" in model_mode:
        target_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
        draft_model = "duckhoi/qwen2.5-7b-eagle3.1-speculator"
    else:
        target_model = prompt_string(
            "Target model ID or path:",
            default_val="Qwen/Qwen2.5-7B-Instruct-AWQ",
        )
        draft_model = prompt_string(
            "Draft speculator ID or path:",
            default_val="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        )

    platform_choice = prompt_choice("Serving platform:", PLATFORM_CHOICES, default_idx=0)
    backend = "modal" if "Modal" in platform_choice else ("colab" if "Colab" in platform_choice else "local")
    target_gpu = "L4"
    if backend == "modal":
        target_gpu = prompt_choice("Modal GPU:", ["L4", "A10G", "T4"], default_idx=0)
    elif backend == "colab":
        target_gpu = "T4"
    preflight = evaluate_serving_preflight(target_model, draft_model, hw_info, backend, target_gpu)
    print(preflight.report.summary())
    preflight.require_safe()
    report = preflight.report

    # Compatibility Report Box (dynamically auto-resizes)
    status_color = "\033[32m" if not report.oom_risk else "\033[31m"
    status_text = "OK" if not report.oom_risk else "OOM RISK"
    vram_detail = f"(~{report.total_vram_needed_mb/1024:.1f} GB needed / {report.free_vram_mb/1024:.1f} GB free)"
    vocab_text = "Shared" if not report.use_heterogeneous_vocab else "Heterogeneous"

    box_items = [
        ("VRAM Status", f"{status_text}  {vram_detail}", status_color),
        ("Speculation", report.draft_method, "\033[36m"),
        ("Vocabulary", vocab_text, "\033[32m"),
    ]
    for warn in report.warnings:
        box_items.append(("Warning", warn, "\033[33m"))

    print("\n" + render_box("Compatibility Report", box_items))

    colab_tunnel_url = ""
    modal_deploy = False
    if backend == "modal":
        if not ModalServingRunner().check_credentials():
            print("Modal credentials missing; use flexsds auth login securely before launch.")
        mode_choice = prompt_choice("Modal deployment mode:",
                                   ["Development (modal serve)", "Production (modal deploy)"], default_idx=0)
        modal_deploy = "Production" in mode_choice

    elif "Google Colab" in platform_choice:
        backend = "colab"
        gpu_choice = prompt_choice("Colab GPU:", ["T4", "A100 / L4"], default_idx=0)
        target_gpu = "T4" if "T4" in gpu_choice else "L4"

        colab_runner = ColabServingRunner()
        instructions = colab_runner.get_colab_instructions(target_gpu=target_gpu)
        print("\n\033[1;37mGoogle Colab Instructions:\033[0m")
        for step in instructions["steps"]:
            print(f"  \033[90m{step}\033[0m")
        print(f"  Notebook URL: \033[34m{instructions['notebook_url']}\033[0m\n")

        colab_tunnel_url = prompt_string("Paste Cloudflare tunnel URL (Enter to skip):", "")
        if colab_tunnel_url:
            print("\033[90mVerifying tunnel connection...\033[0m")
            if colab_runner.verify_tunnel_health(colab_tunnel_url):
                print("\033[32m✔ Cloudflare tunnel is reachable.\033[0m")
            else:
                print("\033[33m⚠ Endpoint not responding yet (model weights may still be loading).\033[0m")

    else:
        backend = "local"
        if not hw_info.is_cuda_available:
            print("\n\033[33m⚠ No NVIDIA CUDA GPU detected on this system.\033[0m")
            print("  Modal or Google Colab is recommended for hardware acceleration.")
        else:
            print(f"\n\033[32m✔ Local GPU: {hw_info.device_name} ({hw_info.free_vram_gb} GB free)\033[0m")

    # Step 5: HardwareAdaptive Optimization Engine
    enable_adaptive = prompt_bool(
        "Enable HardwareAdaptive engine?",
        default_val=True,
    )

    # Step 6: Profile Mapping
    cost_model, profile_name = load_profile_for_hardware(hw_info)
    print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")

    # Save serving configuration
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
    print(f"\033[32m✔ Configuration saved: {config_file.name}\033[0m")

    # Step 7: Launch Server
    should_launch = prompt_bool("Launch server now?", default_val=True)
    if should_launch:
        if backend == "local":
            local_runner = LocalServingRunner()
            run_in_bg = prompt_bool("Run server in background?", default_val=False)
            proc = local_runner.launch(config, background=run_in_bg)
            config["exit_code"] = finish_launch(proc, background=run_in_bg)
            return config
        elif backend == "modal":
            modal_runner = ModalServingRunner()
            proc = modal_runner.launch(config, deploy=modal_deploy)
            config["exit_code"] = finish_launch(proc)
            return config
        elif backend == "colab":
            endpoint = colab_tunnel_url or "http://localhost:8000/v1"
            print("\n\033[1;32mColab guidance only; no server or endpoint created.\033[0m")
            print(f"  \033[1;37mGợi ý:\033[0m Chạy lệnh: \033[36mflexsds chat --endpoint {endpoint}\033[0m để bắt đầu trò chuyện!\n")
    else:
        print("\n\033[1;32mConfiguration saved; no server launched or readiness established.\033[0m")
        print("  \033[1;37mGợi ý:\033[0m Chạy \033[36mflexsds chat\033[0m để kết nối máy chủ khi đã bật.\n")

    return config
