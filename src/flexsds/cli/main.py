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
    from flexsds.core.compatibility import check_compatibility
    from flexsds.core.cost_model import load_profile_for_hardware
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
    from src.flexsds import __version__
    from src.flexsds.cli.banner import print_banner
    from src.flexsds.cli.wizard import run_interactive_wizard
    from src.flexsds.core.hardware import detect_hardware
    from src.flexsds.core.compatibility import check_compatibility
    from src.flexsds.core.cost_model import load_profile_for_hardware
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
        "--no-adaptive", action="store_true", help="Disable TreeAdaptive optimization engine"
    )
    serve_parser.add_argument(
        "--dry-run", action="store_true", help="Print execution command without launching"
    )

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

    # Display banner on interactive startup or regular commands
    print_banner(hw_info)

    if args.command == "detect":
        print(f"\n\033[1;32m✔ Hardware: {hw_info.summary()}\033[0m")
        return 0

    if args.command == "profile":
        _, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[1;32m✔ Linked hardware profile: {profile_name}\033[0m")
        return 0

    if args.command == "serve" and args.target and args.draft:
        # Non-interactive CLI flags mode
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
            runner.launch(cfg, port=args.port, background=args.background, dry_run=args.dry_run)
        elif args.backend == "modal":
            m_runner = ModalServingRunner()
            m_runner.launch(cfg, deploy=args.deploy, dry_run=args.dry_run)
        else:
            if args.dry_run:
                print("  ✔ [Dry-run] Colab notebook URL ready.")
        return 0

    # Default: Interactive serving wizard
    run_interactive_wizard(hw_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
