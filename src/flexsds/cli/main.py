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
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
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

    subparsers = parser.add_subparsers(dest="command", help="Hành động thực thi")

    # Lệnh: detect
    subparsers.add_parser("detect", help="Quét và hiển thị thông số phần cứng GPU")

    # Lệnh: profile
    subparsers.add_parser("profile", help="Đo đạc hồ sơ chi phí phần cứng GPU")

    # Lệnh: stop
    subparsers.add_parser("stop", help="Dừng máy chủ FlexSDS đang chạy")

    # Lệnh: serve
    serve_parser = subparsers.add_parser("serve", help="Khởi chạy phục vụ mô hình")
    serve_parser.add_argument("--target", type=str, help="Tên mô hình đích")
    serve_parser.add_argument("--draft", type=str, help="Tên mô hình nháp")
    serve_parser.add_argument(
        "--backend", choices=["modal", "local", "colab"], default="modal", help="Nền tảng phục vụ"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Cổng mạng phục vụ API"
    )
    serve_parser.add_argument(
        "--background", action="store_true", help="Chạy máy chủ ở chế độ nền"
    )
    serve_parser.add_argument(
        "--deploy", action="store_true", help="Triển khai cố định trên Modal Cloud"
    )
    serve_parser.add_argument(
        "--no-adaptive", action="store_true", help="Tắt lớp thích ứng TreeAdaptive"
    )
    serve_parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ xuất lệnh phục vụ, không khởi chạy"
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Điểm vào chính của giao diện dòng lệnh FlexSDS."""
    args = parse_args(argv)
    hw_info = detect_hardware()

    if args.command == "stop":
        local_runner = LocalServingRunner()
        local_stopped = local_runner.stop()
        if local_stopped:
            print("\033[32m✔ Đã dừng tiến trình máy chủ vLLM cục bộ thành công.\033[0m")
        else:
            print("\033[90mKhông tìm thấy tiến trình máy chủ cục bộ đang chạy (hoặc đã dừng).\033[0m")
        return 0

    # Hiển thị banner khi khởi chạy tương tác hoặc theo lệnh thông thường
    print_banner(hw_info)

    if args.command == "detect":
        print(f"\n\033[1;32m✔ Trạng thái phần cứng: {hw_info.summary()}\033[0m")
        return 0

    if args.command == "profile":
        _, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[1;32m✔ Đã liên kết với hồ sơ phần cứng: {profile_name}\033[0m")
        return 0

    if args.command == "serve" and args.target and args.draft:
        # Chế độ non-interactive qua CLI flags
        print(f"\n\033[1;33mKhởi chạy phục vụ (CLI Mode): Target={args.target}, Draft={args.draft}, Backend={args.backend}\033[0m")
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
                print("  ✔ [Dry-run] Colab notebook URL sẵn sàng.")
        return 0

    # Mặc định: Chạy giao diện hỏi-đáp tương tác (Wizard)
    run_interactive_wizard(hw_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
