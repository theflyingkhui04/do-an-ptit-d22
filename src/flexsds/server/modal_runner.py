"""Modal cloud serving runner for FlexSDS (BYOC, ephemeral, and persistent deploy)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flexsds.core.credentials import CredentialsManager
except ModuleNotFoundError:
    from src.flexsds.core.credentials import CredentialsManager


class ModalServingRunner:
    """Quản lý triển khai FlexSDS Serverless trên nền tảng đám mây Modal.com."""

    DEFAULT_APP_NAME = "flexsds-serving"

    def __init__(self, credentials_manager: Optional[CredentialsManager] = None) -> None:
        self.cred_mgr = credentials_manager or CredentialsManager()

    def check_credentials(self) -> bool:
        """Kiểm tra thông tin định danh Modal qua CredentialsManager, biến môi trường hoặc ~/.modal.toml."""
        self.cred_mgr.inject_into_environ()

        token_id = os.environ.get("MODAL_TOKEN_ID") or self.cred_mgr.get_credential("modal", "MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET") or self.cred_mgr.get_credential("modal", "MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            return True

        toml_path = os.path.expanduser("~/.modal.toml")
        if os.path.exists(toml_path):
            return True

        return False

    def build_modal_command(self, config: Dict[str, Any], deploy: bool = False) -> List[str]:
        """Tạo danh sách đối số dòng lệnh gọi công cụ CLI modal tới modal_app.py."""
        serving_cfg = config.get("serving", config)
        target_gpu = serving_cfg.get("target_gpu", "L4")

        app_path = str(Path(__file__).parent / "modal_app.py")
        subcommand = "deploy" if deploy else "run"
        cmd = [
            "modal",
            subcommand,
            app_path,
            "--gpu",
            target_gpu,
        ]
        return cmd

    def launch(
        self,
        config: Dict[str, Any],
        deploy: bool = False,
        dry_run: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Khởi chạy ứng dụng Modal serverless."""
        if not self.check_credentials() and not dry_run:
            print("\n\033[31m[Lỗi định danh]\033[0m Chưa cấu hình thông tin xác thực Modal.")
            print("  Vui lòng chạy: flexsds auth login --provider modal")
            print("  Hoặc cấu hình qua lệnh: modal setup")
            return None

        cmd = self.build_modal_command(config, deploy=deploy)

        if dry_run:
            mode_desc = "Persistent Deploy" if deploy else "Ephemeral Run"
            print(f"\n[FlexSDS Modal] Dry-run command ({mode_desc}):")
            print(" ".join(cmd))
            return None

        try:
            proc = subprocess.Popen(cmd)
            return proc
        except FileNotFoundError:
            print("\n\033[31m[Lỗi]\033[0m Không tìm thấy công cụ CLI 'modal'. Cài đặt: pip install modal")
            return None

    def check_process_status(self, proc: subprocess.Popen) -> tuple[bool, Optional[int]]:
        """Kiểm tra trạng thái tiến trình Modal; phát hiện sớm nếu ứng dụng dừng bất ngờ."""
        code = proc.poll()
        if code is None:
            return True, None
        return False, code

    def stop(self, app_name: str = DEFAULT_APP_NAME) -> bool:
        """Dừng ứng dụng đang chạy trên hạ tầng Modal Cloud."""
        cmd = ["modal", "app", "stop", app_name]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False
