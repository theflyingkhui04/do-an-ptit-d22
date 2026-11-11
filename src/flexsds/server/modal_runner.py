"""Modal cloud serving runner for FlexSDS (BYOC, ephemeral, and persistent deploy)."""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional


class ModalServingRunner:
    """Quản lý triển khai FlexSDS Serverless trên nền tảng đám mây Modal.com."""

    DEFAULT_APP_NAME = "ptit-vllm"

    def check_credentials(self) -> bool:
        """Kiểm tra thông tin định danh Modal (qua biến môi trường hoặc ~/.modal.toml)."""
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            return True

        toml_path = os.path.expanduser("~/.modal.toml")
        if os.path.exists(toml_path):
            return True

        return False

    def build_modal_command(self, config: Dict[str, Any], deploy: bool = False) -> List[str]:
        """Tạo danh sách đối số dòng lệnh gọi công cụ CLI modal."""
        serving_cfg = config.get("serving", config)
        target_gpu = serving_cfg.get("target_gpu", "L4")

        subcommand = "deploy" if deploy else "run"
        cmd = [
            "modal",
            subcommand,
            "experiments/modal_lanes.py",
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
        cmd = self.build_modal_command(config, deploy=deploy)

        if dry_run:
            mode_desc = "Cố định (Persistent Deploy)" if deploy else "Tạm thời (Ephemeral Run)"
            print(f"\n[FlexSDS Modal] Lệnh thực thi ({mode_desc} - Dry-run):")
            print(" ".join(cmd))
            return None

        return subprocess.Popen(cmd)

    def stop(self, app_name: str = DEFAULT_APP_NAME) -> bool:
        """Dừng ứng dụng đang chạy trên hạ tầng Modal Cloud."""
        cmd = ["modal", "app", "stop", app_name]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False
