"""FlexSDS CLI main dispatcher module."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    from flexsds import __version__
    from flexsds.cli.banner import print_banner
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import detect_hardware
    from flexsds.core.compatibility import check_compatibility
    from flexsds.core.cost_model import load_profile_for_hardware
    from flexsds.core.credentials import CredentialsManager
    from flexsds.client.chat import run_interactive_chat
    from flexsds.client.monitor import run_live_monitor
    from flexsds.cli.task_runner import TaskRunner
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds import __version__
    from src.flexsds.cli.banner import print_banner
    from src.flexsds.cli.wizard import run_interactive_wizard
    from src.flexsds.core.hardware import detect_hardware
    from src.flexsds.core.compatibility import check_compatibility
    from src.flexsds.core.cost_model import load_profile_for_hardware
    from src.flexsds.core.credentials import CredentialsManager
    from src.flexsds.client.chat import run_interactive_chat
    from src.flexsds.client.monitor import run_live_monitor
    from src.flexsds.cli.task_runner import TaskRunner
    from src.flexsds.server.local_runner import LocalServingRunner
    from src.flexsds.server.modal_runner import ModalServingRunner


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flexsds",
        description="FlexSDS: Flexible Speculative Decoding Serving Engine for LLMs",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"flexsds {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: detect
    subparsers.add_parser("detect", help="Detect and display hardware / GPU specs")

    # Command: profile
    subparsers.add_parser("profile", help="Load hardware cost profile for current GPU")

    # Command: stop
    subparsers.add_parser("stop", help="Stop running local FlexSDS server")

    # Command: auth
    auth_parser = subparsers.add_parser("auth", help="Manage secure credentials outside repository")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action", help="Auth action to perform")
    
    # auth login
    auth_login_parser = auth_subparsers.add_parser("login", help="Save API key / token securely")
    auth_login_parser.add_argument("--provider", choices=["modal", "huggingface", "cloudflare"], default="modal", help="Target service provider")
    auth_login_parser.add_argument("--key", type=str, default=None, help="Specific credential key name")
    auth_login_parser.add_argument("--token", type=str, default=None, help="Token value (prompted securely if omitted)")

    # auth status
    auth_subparsers.add_parser("status", help="Show masked status of saved credentials")

    # auth logout
    auth_logout_parser = auth_subparsers.add_parser("logout", help="Delete saved credentials")
    auth_logout_parser.add_argument("--provider", type=str, default=None, help="Target service provider to remove")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch speculative serving server")
    serve_parser.add_argument("--target", type=str, help="Target model identifier")
    serve_parser.add_argument("--draft", type=str, help="Draft model or speculator head identifier")
    serve_parser.add_argument(
        "--backend", choices=["modal", "local", "colab"], default="modal", help="Serving backend platform"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="API server port"
    )
    serve_parser.add_argument(
        "--background", action="store_true", help="Run local server as background daemon"
    )
    serve_parser.add_argument(
        "--deploy", action="store_true", help="Deploy persistently on Modal Cloud"
    )
    serve_parser.add_argument(
        "--no-adaptive", action="store_true", help="Disable HardwareAdaptive optimization engine"
    )
    serve_parser.add_argument(
        "--dry-run", action="store_true", help="Print execution command without launching"
    )

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive streaming chat with served model")
    chat_parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1", help="API base URL")
    chat_parser.add_argument("--model", type=str, default=None, help="Served model name override")
    chat_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    # Command: monitor
    monitor_parser = subparsers.add_parser("monitor", help="Live telemetry monitor for speculative decoding traces")
    monitor_parser.add_argument("--trace-file", type=str, default="telemetry/serving_trace.jsonl", help="Path to JSONL telemetry trace")
    monitor_parser.add_argument("--refresh-rate", type=float, default=1.0, help="Telemetry poll interval in seconds")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for FlexSDS command-line interface."""
    args = parse_args(argv)
    hw_info = detect_hardware()

    if args.command == "stop":
        local_runner = LocalServingRunner()
        local_stopped = local_runner.stop()
        if local_stopped:
            print("\033[32m✔ Local vLLM server process stopped successfully.\033[0m")
        else:
            print("\033[90mNo active local server process found.\033[0m")
        return 0

    if args.command == "auth":
        cred_mgr = CredentialsManager()
        if args.auth_action == "status":
            masked = cred_mgr.get_all_masked()
            print("\n\033[1;37mFlexSDS Secure Credentials (~/.flexsds/credentials):\033[0m")
            if not masked:
                print("  \033[90m(No credentials saved. Run 'flexsds auth login' to add.)\033[0m")
            for prov, keys in masked.items():
                print(f"  \033[32m✔ {prov}:\033[0m")
                for k, v in keys.items():
                    print(f"      {k}: {v}")
            return 0
        elif args.auth_action == "logout":
            target_prov = getattr(args, "provider", None)
            if target_prov:
                deleted = cred_mgr.delete_credential(target_prov)
                if deleted:
                    print(f"\033[32m✔ Removed credentials for provider: {target_prov}\033[0m")
                else:
                    print(f"\033[90mNo credentials found for provider: {target_prov}\033[0m")
            else:
                for prov in list(cred_mgr.get_all_masked().keys()):
                    cred_mgr.delete_credential(prov)
                print("\033[32m✔ Removed all stored credentials.\033[0m")
            return 0
        elif args.auth_action == "login":
            import getpass
            prov = args.provider
            key_name = args.key or ("HF_TOKEN" if prov == "huggingface" else ("MODAL_TOKEN_ID" if prov == "modal" else "TOKEN"))
            val = args.token or getpass.getpass(f"Enter {key_name} for {prov}: ")
            if val:
                cred_mgr.set_credential(prov, key_name, val.strip())
                print(f"\033[32m✔ Securely saved {key_name} for {prov} with mode 0600.\033[0m")
            return 0
        return 0

    # Display banner on interactive startup or regular commands
    print_banner(hw_info)

    if args.command == "detect":
        print(f"\n\033[1;32m✔ Hardware: {hw_info.summary()}\033[0m")
        return 0

    if args.command == "profile":
        _, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[1;32m✔ Linked hardware profile: {profile_name}\033[0m")
        return 0

    if args.command == "chat":
        return run_interactive_chat(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
        )

    if args.command == "monitor":
        return run_live_monitor(
            trace_file=args.trace_file,
            refresh_rate=args.refresh_rate,
        )

    if args.command == "serve" and args.target and args.draft:
        if args.dry_run:
            # Chế độ kiểm tra lệnh dry-run
            print(f"\n\033[1;37mLaunching server: Target={args.target}, Draft={args.draft}, Backend={args.backend}\033[0m")
            report = check_compatibility(args.target, args.draft, free_vram_mb=hw_info.free_vram_mb)
            print(f"  ✔ {report.summary()}")

            cfg = {
                "serving": {
                    "target_model": args.target,
                    "draft_model": args.draft,
                    "draft_method": report.draft_method,
                    "use_heterogeneous_vocab": report.use_heterogeneous_vocab,
                    "port": args.port,
                    "enable_adaptive": not args.no_adaptive,
                }
            }

            if args.backend == "local":
                runner = LocalServingRunner()
                runner.launch(cfg, port=args.port, background=args.background, dry_run=True)
            elif args.backend == "modal":
                m_runner = ModalServingRunner()
                m_runner.launch(cfg, deploy=args.deploy, dry_run=True)
            else:
                print("  ✔ [Dry-run] Colab notebook URL ready.")
            return 0

        # Chế độ phục vụ thực tế: Điều phối bằng TaskRunner với cơ chế thu gọn log và phím Ctrl+O
        task_runner = TaskRunner()
        t1 = task_runner.add_task("Kiểm tra tính tương thích mô hình và phân bổ VRAM")
        t2 = task_runner.add_task("Cấu hình điều phối suy luận HardwareAdaptive")
        t3 = task_runner.add_task(f"Khởi chạy máy chủ phục vụ ({args.backend})")

        def step1_action(t):
            t.log(f"Mô hình mục tiêu: {args.target}")
            t.log(f"Mô hình nháp: {args.draft}")
            rep = check_compatibility(args.target, args.draft, free_vram_mb=hw_info.free_vram_mb)
            t.log(f"Kết quả kiểm tra: {rep.summary()}")
            return rep

        report = task_runner.run_task(t1, step1_action)

        cfg = {
            "serving": {
                "target_model": args.target,
                "draft_model": args.draft,
                "draft_method": report.draft_method,
                "use_heterogeneous_vocab": report.use_heterogeneous_vocab,
                "port": args.port,
                "enable_adaptive": not args.no_adaptive,
            }
        }

        def step2_action(t):
            _, profile_name = load_profile_for_hardware(hw_info)
            t.log(f"Hồ sơ phần cứng: {profile_name}")
            t.log(f"HardwareAdaptive: {'bật' if not args.no_adaptive else 'tắt'}")
            return profile_name

        task_runner.run_task(t2, step2_action)

        def step3_action(t):
            if args.backend == "local":
                runner = LocalServingRunner()
                t.log(f"Khởi chạy máy chủ vLLM cục bộ trên cổng {args.port}...")
                return runner.launch(cfg, port=args.port, background=args.background, dry_run=False)
            elif args.backend == "modal":
                m_runner = ModalServingRunner()
                t.log("Khởi tạo phiên chạy Modal Serverless...")
                return m_runner.launch(cfg, deploy=args.deploy, dry_run=False)
            else:
                t.log("Cung cấp chỉ dẫn kết nối Google Colab.")
                return 0

        task_runner.run_task(t3, step3_action)
        print("\n\033[1;32m✔ Tất cả tác vụ phục vụ đã hoàn tất!\033[0m")
        print("  \033[1;37mGợi ý:\033[0m Chạy \033[36mflexsds chat\033[0m để bắt đầu trò chuyện hoặc \033[36mflexsds monitor\033[0m để theo dõi thích ứng.\n")
        return 0

    # Default: Interactive serving wizard
    run_interactive_wizard(hw_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
