"""Modal cloud serving runner for FlexSDS (BYOC, ephemeral, and persistent deploy)."""
from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flexsds.cli.output import print_output as print
    from flexsds.core.credentials import CredentialsManager
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
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
        app_path = str(Path(__file__).parent / "modal_app.py")
        return ["modal", "deploy" if deploy else "serve", app_path]

    def setup_environment(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Forward non-secret serving settings and package paths for app construction."""
        cfg = config.get("serving", config)
        keys = ("target_model", "draft_model", "draft_method", "use_heterogeneous_vocab",
                "port", "enable_adaptive", "target_gpu", "max_model_len", "max_num_seqs")
        env = dict(os.environ)
        env["FLEXSDS_SERVING_CONFIG"] = json.dumps({k: cfg[k] for k in keys if k in cfg})

        # Inject package root and sys.path entries into PYTHONPATH so modal subprocess can import flexsds
        pkg_root = str(Path(__file__).resolve().parents[2])
        existing_pp = env.get("PYTHONPATH", "")
        extra_paths = [pkg_root]
        for p in sys.path:
            if p and p not in extra_paths and os.path.isdir(p):
                extra_paths.append(p)
        if existing_pp:
            extra_paths.append(existing_pp)
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)
        return env

    def launch(
        self,
        config: Dict[str, Any],
        deploy: bool = False,
        dry_run: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Khởi chạy ứng dụng Modal serverless."""
        if not dry_run and not self.check_credentials():
            print("\n\033[31m[Authentication Error]\033[0m Modal credentials not configured.")
            print("  Please run: flexsds auth login --provider modal")
            print("  Or configure via: modal setup")
            return None

        cmd = self.build_modal_command(config, deploy=deploy)

        if dry_run:
            mode_desc = "Persistent Deploy" if deploy else "Ephemeral Run"
            print(f"\n[FlexSDS Modal] Dry-run command ({mode_desc}):")
            print(" ".join(cmd))
            print(f"Serving configuration: {self.setup_environment(config)['FLEXSDS_SERVING_CONFIG']}")
            return None

        try:
            proc = subprocess.Popen(cmd, env=self.setup_environment(config))
            return proc
        except FileNotFoundError:
            print("\n\033[31m[Error]\033[0m 'modal' CLI not found. Install with: pip install modal")
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
