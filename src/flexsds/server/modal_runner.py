"""Modal cloud serving runner for FlexSDS (BYOC, ephemeral, and persistent deploy)."""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import re
import time
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
        from flexsds.server.modal_auth import prepare_modal_auth
        return prepare_modal_auth(self.cred_mgr)

    def build_modal_command(self, config: Dict[str, Any], deploy: bool = False) -> List[str]:
        """Tạo danh sách đối số dòng lệnh gọi công cụ CLI modal tới modal_app.py."""
        modal_bin = shutil.which("modal") or "modal"
        app_path = str(Path(__file__).parent / "modal_app.py")
        return [modal_bin, "deploy" if deploy else "serve", app_path]

    def setup_environment(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Forward non-secret serving settings and package paths for app construction."""
        cfg = config.get("serving", config)
        keys = ("target_model", "draft_model", "draft_method", "use_heterogeneous_vocab",
                "target_revision", "draft_revision", "port", "enable_adaptive", "target_gpu", "max_model_len", "max_num_seqs", "profile_path")
        if (self.cred_mgr.path.parent / 'modal_logged_out').exists():
            # Calibration also uses this method without the serving launch preflight.
            from flexsds.server.modal_auth import prepare_modal_auth
            prepare_modal_auth(self.cred_mgr)
        env = dict(os.environ)
        # Redirected Python output on Windows otherwise uses the ANSI code page.
        # Modal emits Unicode status symbols, so both ends of the pipe use UTF-8.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["FLEXSDS_SERVING_CONFIG"] = json.dumps({k: cfg[k] for k in keys if k in cfg})

        # Modal can use another pipx interpreter; forward project source only,
        # never the launcher's third-party packages or interpreter directories.
        package_root = Path(__file__).resolve().parents[2]
        env['PYTHONPATH'] = os.pathsep.join((str(package_root), str(package_root.parent)))
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

        cfg = config.get('serving', config)
        if cfg.get('enable_adaptive', False):
            from flexsds.server.local_runner import LocalServingRunner
            LocalServingRunner().build_command(config)
            LocalServingRunner().setup_environment(config)

        cmd = self.build_modal_command(config, deploy=deploy)

        if dry_run:
            mode_desc = "Persistent Deploy" if deploy else "Ephemeral Run"
            print(f"\n[FlexSDS Modal] Dry-run command ({mode_desc}):")
            print(" ".join(cmd))
            print(f"Serving configuration: {self.setup_environment(config)['FLEXSDS_SERVING_CONFIG']}")
            return None

        try:
            from flexsds.cli.process_progress import ProcessProgress
            output = ProcessProgress('serving')
            proc = None
            try:
                proc = subprocess.Popen(cmd, env=self.setup_environment(config), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace', bufsize=1)
                output.start()
            except BaseException:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                output.close()
                raise
            endpoint = []
            def observe_endpoint(line):
                match = re.search(r"https://[a-zA-Z0-9.-]+\.modal\.run", line)
                if match and not endpoint:
                    endpoint.append(match.group())
            output.capture(proc, on_line=observe_endpoint, close_on_exit=not deploy)
            proc._flexsds_output = output
            logs_done = output.done
            def ready():
                from flexsds.server.readiness import verify_endpoint
                deadline = time.monotonic() + 900
                while not endpoint:
                    code = proc.poll()
                    if code is not None and logs_done.is_set() and not endpoint:
                        raise RuntimeError(f"Modal exited before exposing endpoint (exit {code})")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Timed out waiting for Modal endpoint URL")
                    time.sleep(0.2)
                output.notice(f"Endpoint: {endpoint[0]}")
                output.update('Waiting for endpoint readiness',15)
                model = verify_endpoint(endpoint[0], process=proc, deployed=deploy,
                                        timeout=max(0.1, deadline-time.monotonic()))
                proc._flexsds_session = dict(endpoint=endpoint[0].rstrip('/')+'/v1', model=model,
                                            config=cfg, deploy=deploy)
                output.notice(f"Inference verified: {endpoint[0]}/v1 (model={model})")
                output.update('Serving ready' if not deploy else 'Deployment verified',100)
                if deploy:
                    output.close()
            proc._flexsds_readiness = ready
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
        modal_bin = shutil.which("modal") or "modal"
        cmd = [modal_bin, "app", "stop", app_name]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False
