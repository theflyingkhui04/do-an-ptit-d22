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
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    from flexsds import __version__
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
    from src.flexsds import __version__
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
    "Modal.com (Cloud Serverless GPU)",
    "Google Colab (Free GPU T4/L4)",
    "Local vLLM (Self-hosted GPU)",
]

MODAL_GPU_CHOICES = [
    "A10G (Recommended - 24 GB VRAM)",
    "L4 (24 GB VRAM - Capacity limited)",
    "T4 (16 GB VRAM)",
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


def _get_key_stroke() -> str:
    """Reads a single key stroke across Linux/macOS and Windows."""
    if HAS_MSVCRT:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            elif ch2 == "P":
                return "down"
            return "other"
        elif ch in ("\r", "\n", " "):
            return "enter"
        elif ch == "\x03":
            return "ctrl_c"
        elif ch.isdigit():
            return f"digit:{ch}"
        elif ch in ("k", "K"):
            return "up"
        elif ch in ("j", "J"):
            return "down"
        return "other"

    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    elif ch3 == "B":
                        return "down"
                return "other"
            elif ch in ("\r", "\n", " "):
                return "enter"
            elif ch == "\x03":
                return "ctrl_c"
            elif ch.isdigit():
                return f"digit:{ch}"
            elif ch in ("k", "K"):
                return "up"
            elif ch in ("j", "J"):
                return "down"
            return "other"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return "fallback"


def prompt_choice(question: str, choices: list[str], default_idx: int = 0) -> str:
    """Prompt user to select an option using arrow keys and Enter."""
    if not sys.stdin.isatty():
        print(f"\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{choices[default_idx]}\033[0m")
        return choices[default_idx]

    # If neither termios nor msvcrt is available, use interactive numbered input
    if not HAS_TERMIOS and not HAS_MSVCRT:
        print(f"\n\033[1;37m? {question}\033[0m")
        for i, c in enumerate(choices, 1):
            marker = " (default)" if i - 1 == default_idx else ""
            print(f"  \033[90m{i}.\033[0m {c}{marker}")
        try:
            raw = input(f"Select [1-{len(choices)}] (default {default_idx + 1}): ").strip()
            if not raw:
                choice = choices[default_idx]
            elif raw.isdigit() and 1 <= int(raw) <= len(choices):
                choice = choices[int(raw) - 1]
            else:
                choice = choices[default_idx]
        except (EOFError, KeyboardInterrupt):
            print("\n")
            sys.exit(0)
        print(f"\033[32m✔\033[0m \033[1m{question}\033[0m \033[36m{choice}\033[0m")
        return choice

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

    try:
        while True:
            key = _get_key_stroke()
            if key == "up":
                current = (current - 1) % n
                render()
            elif key == "down":
                current = (current + 1) % n
                render()
            elif key.startswith("digit:"):
                idx = int(key.split(":")[1]) - 1
                if 0 <= idx < n:
                    current = idx
                    render()
                    break
            elif key == "enter":
                break
            elif key == "ctrl_c":
                print("\033[?25h\n", end="")
                sys.exit(0)
    finally:
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
    sep = " " if question.rstrip().endswith(":") else ": "
    try:
        print(f"\n\033[1;37m? {question}\033[0m{hint}{sep}", end="")
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
    target_gpu = "A10G"
    if backend == "modal":
        modal_gpu_choices = [
            "A10G (Recommended - 24 GB VRAM)",
            "L4 (24 GB VRAM - Capacity limited)",
            "T4 (16 GB VRAM)",
        ]
        chosen = prompt_choice("Modal GPU:", modal_gpu_choices, default_idx=0)
        target_gpu = "A10G" if "A10G" in chosen else ("L4" if "L4" in chosen else "T4")
    elif backend == "colab":
        gpu_choice = prompt_choice("Colab GPU:", ["T4", "A100 / L4"], default_idx=0)
        target_gpu = "T4" if "T4" in gpu_choice else "L4"
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
        modal_runner = ModalServingRunner()
        if not modal_runner.check_credentials():
            print("\n\033[1;33m⚠ [Authentication Warning] Modal credentials not configured.\033[0m")
            print("  Modal requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to serve or deploy.")
            setup_now = prompt_bool("Configure Modal credentials now?", default_val=True)
            if setup_now:
                token_id = prompt_string("Enter MODAL_TOKEN_ID (or press Enter to skip):", default_val="")
                token_secret = prompt_string("Enter MODAL_TOKEN_SECRET (or press Enter to skip):", default_val="")
                if token_id and token_secret:
                    modal_runner.cred_mgr.set_credential("modal", "MODAL_TOKEN_ID", token_id.strip())
                    modal_runner.cred_mgr.set_credential("modal", "MODAL_TOKEN_SECRET", token_secret.strip())
                    modal_runner.cred_mgr.inject_into_environ()
                    print("\033[32m✔ Modal credentials saved securely.\033[0m")
                else:
                    print("  Credentials not provided. You can configure later via 'flexsds auth login --provider modal'.")
            else:
                print("  Skipped. You can configure later via 'flexsds auth login --provider modal' or 'modal setup'.")

        mode_choice = prompt_choice("Modal deployment mode:",
                                   ["Development (modal serve)", "Production (modal deploy)"], default_idx=0)
        modal_deploy = "Production" in mode_choice

    elif backend == "colab":
        colab_runner = ColabServingRunner()
        nb_path = colab_runner.resolve_notebook_path()
        colab_runner.sync_notebook_config({
            "target_model": target_model,
            "draft_model": draft_model,
            "draft_method": report.draft_method,
            "enable_adaptive": True,
        }, notebook_path=str(nb_path))

        instructions = colab_runner.get_colab_instructions(target_gpu=target_gpu)
        abs_nb = str(nb_path.resolve())
        parent_dir = str(nb_path.parent.resolve())
        cmd_linux = f'xdg-open "{parent_dir}"'
        cmd_win = f'explorer /select,"{abs_nb}"'

        print("\n\033[1;37mGoogle Colab Instructions:\033[0m")
        for step in instructions["steps"]:
            print(f"  \033[90m{step}\033[0m")
        print(f"\n  \033[1;37mLocal notebook path (upload this to Colab):\033[0m")
        print(f"    \033[36m{abs_nb}\033[0m")
        print(f"  \033[1;37mCommands to reveal notebook folder:\033[0m")
        print(f"    • Linux  : \033[33m{cmd_linux}\033[0m")
        print(f"    • Windows: \033[33m{cmd_win}\033[0m")
        print(f"  \033[1;37mDirect Colab URL (if repository is authorized):\033[0m")
        print(f"    \033[34m{instructions['notebook_url']}\033[0m\n")

        open_folder = prompt_bool("Open containing folder in file manager now?", default_val=False)
        if open_folder:
            if colab_runner.open_in_file_manager(nb_path):
                print("  \033[32m✔ File manager opened.\033[0m")
            else:
                print("  \033[33m⚠ Could not open file manager automatically. Please use the command above.\033[0m")

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
        "flexsds_version": __version__,
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
            if not modal_runner.check_credentials():
                print("\n\033[1;31m✖ [Authentication Error] Modal credentials not configured.\033[0m")
                print("  Please run: flexsds auth login --provider modal")
                print("  Or configure via: modal setup")
                config["exit_code"] = 1
                return config
            proc = modal_runner.launch(config, deploy=modal_deploy)
            config["exit_code"] = finish_launch(proc)
            return config
        elif backend == "colab":
            if colab_tunnel_url:
                clean_tunnel = colab_tunnel_url.rstrip("/")
                if not clean_tunnel.endswith("/v1"):
                    endpoint = f"{clean_tunnel}/v1"
                else:
                    endpoint = clean_tunnel
            else:
                endpoint = "http://localhost:8000/v1"
            config["endpoint"] = endpoint
            print("\n\033[1;32mColab guidance only; no server or endpoint created.\033[0m")
            print(f"  \033[1;37mTip:\033[0m Run command: \033[36mflexsds chat --endpoint {endpoint}\033[0m to start chatting!\n")
    else:
        print("\n\033[1;32mConfiguration saved; no server launched or readiness established.\033[0m")
        print("  \033[1;37mTip:\033[0m Run \033[36mflexsds chat\033[0m to connect when server is running.\n")

    return config
