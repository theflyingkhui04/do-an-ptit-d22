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
except ModuleNotFoundError:
    from src.flexsds import __version__
    from src.flexsds.cli.banner import print_banner
    from src.flexsds.cli.wizard import run_interactive_wizard
    from src.flexsds.core.hardware import detect_hardware
    from src.flexsds.core.compatibility import check_compatibility
    from src.flexsds.core.cost_model import load_profile_for_hardware


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flexsds",
        description="FlexSDS: Flexible Speculative Decoding Serving Engine for LLMs",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"flexsds {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Hành động thực thi")

    # Command: detect
    subparsers.add_parser("detect", help="Quét và hiển thị thông số phần cứng GPU")

    # Command: profile
    subparsers.add_parser("profile", help="Đo đạc hồ sơ chi phí phần cứng GPU")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Khởi chạy phục vụ mô hình")
    serve_parser.add_argument("--target", type=str, help="Tên mô hình đích")
    serve_parser.add_argument("--draft", type=str, help="Tên mô hình nháp")
    serve_parser.add_argument(
        "--backend", choices=["modal", "local"], default="modal", help="Nền tảng phục vụ"
    )
    serve_parser.add_argument(
        "--no-adaptive", action="store_true", help="Tắt lớp thích ứng TreeAdaptive"
    )
    serve_parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ xuất lệnh phục vụ, không khởi chạy"
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entrypoint for flexsds."""
    args = parse_args(argv)
    hw_info = detect_hardware()

    # Always show banner on interactive start or when explicitly requested
    print_banner(hw_info)

    if args.command == "detect":
        print(f"\n\033[1;32m✔ Trạng thái phần cứng: {hw_info.summary()}\033[0m")
        return 0

    if args.command == "profile":
        _, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[1;32m✔ Đã liên kết với hồ sơ phần cứng: {profile_name}\033[0m")
        return 0

    if args.command == "serve" and args.target and args.draft:
        # Non-interactive CLI flags mode
        print(f"\n\033[1;33mKhởi chạy phục vụ (CLI Mode): Target={args.target}, Draft={args.draft}, Backend={args.backend}\033[0m")
        report = check_compatibility(args.target, args.draft, free_vram_mb=hw_info.free_vram_mb)
        print(f"  ✔ {report.summary()}")
        if args.dry_run:
            print(f"  ✔ [Dry-run] Cấu hình hợp lệ. Sẵn sàng triển khai trên {args.backend}.")
        return 0

    # Default: Run interactive wizard
    run_interactive_wizard(hw_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
