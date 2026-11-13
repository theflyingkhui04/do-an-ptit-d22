"""FlexSDS CLI main dispatcher module."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    from flexsds.cli.output import print_output as print
    from flexsds import __version__
    from flexsds.cli.banner import print_banner
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import detect_hardware
    from flexsds.core.compatibility import evaluate_serving_preflight, ServingBlockedError
    from flexsds.core.cost_model import load_profile_for_hardware
    from flexsds.core.credentials import CredentialsManager
    from flexsds.client.chat import run_interactive_chat
    from flexsds.client.monitor import run_live_monitor
    from flexsds.cli.task_runner import TaskRunner
    from flexsds.server.local_runner import LocalServingRunner, finish_launch
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
    from src.flexsds import __version__
    from src.flexsds.cli.banner import print_banner
    from src.flexsds.cli.wizard import run_interactive_wizard
    from src.flexsds.core.hardware import detect_hardware
    from src.flexsds.core.compatibility import evaluate_serving_preflight, ServingBlockedError
    from src.flexsds.core.cost_model import load_profile_for_hardware
    from src.flexsds.core.credentials import CredentialsManager
    from src.flexsds.client.chat import run_interactive_chat
    from src.flexsds.client.monitor import run_live_monitor
    from src.flexsds.cli.task_runner import TaskRunner
    from src.flexsds.server.local_runner import LocalServingRunner, finish_launch
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
    serve_parser.add_argument("--gpu", choices=["L4", "T4", "A10G"], default="L4", help="Remote GPU memory budget and Modal allocation")
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
    if args.command == "serve" and not 1 <= args.port <= 65535:
        print("Invalid port: expected 1..65535")
        return 2
    if args.command == "monitor" and args.refresh_rate <= 0:
        print("Refresh rate must be positive")
        return 2
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
                print(f"\033[32m✔ Securely saved {key_name} for {prov} (POSIX mode 0600; Windows uses filesystem ACLs).\033[0m")
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
        preflight = evaluate_serving_preflight(args.target, args.draft, hw_info,
                                               args.backend, args.gpu)
        print(preflight.report.summary())
        if not preflight.is_safe:
            print("Launch blocked: " + "; ".join(preflight.reasons))
        if args.backend != "local":
            print(f"Remote {args.gpu} memory budget is estimated, not measured free VRAM.")
        cfg = {"serving": {
            "target_model": args.target, "draft_model": args.draft,
            "draft_method": preflight.report.draft_method,
            "use_heterogeneous_vocab": preflight.report.use_heterogeneous_vocab,
            "port": args.port, "target_gpu": args.gpu,
            "enable_adaptive": not args.no_adaptive,
        }}
        if args.backend == "colab":
            print("Colab guidance only: no server or endpoint was created; supplied models are not forwarded to the notebook.")
            print("Open notebooks/flexsds_colab_server.ipynb and configure the exact models on that runtime.")
            return 0
        if not args.dry_run and not preflight.is_safe:
            return 2
        if args.dry_run:
            print("Preview only; not a readiness check.")
            if args.backend == "local":
                LocalServingRunner().launch(cfg, port=args.port, background=args.background, dry_run=True)
            else:
                ModalServingRunner().launch(cfg, deploy=args.deploy, dry_run=True)
            return 0
        tasks = TaskRunner()
        task = tasks.add_task(f"Start {args.backend} serving process")
        def launch(t):
            if args.backend == "local":
                return LocalServingRunner().launch(cfg, port=args.port, background=args.background)
            return ModalServingRunner().launch(cfg, deploy=args.deploy)
        proc = tasks.run_task(task, launch)
        return finish_launch(proc, background=args.backend == "local" and args.background)

    try:
        result = run_interactive_wizard(hw_info)
        return result.get("exit_code", 0) if isinstance(result, dict) else 0
    except ServingBlockedError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
