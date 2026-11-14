"""Modal HTTP app; configuration is passed by the CLI, never embedded credentials.

Development: modal serve. Persistent service: modal deploy.
The web_server decorator owns startup readiness and endpoint exposure.
"""
from __future__ import annotations
import json
import os
import sys
import subprocess
from pathlib import Path

# Ensure flexsds package root is in sys.path when executed directly by modal CLI
_pkg_root = str(Path(__file__).resolve().parents[2])
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    from flexsds.server.local_runner import LocalServingRunner
except ModuleNotFoundError:
    try:
        from src.flexsds.server.local_runner import LocalServingRunner
    except ModuleNotFoundError:
        from local_runner import LocalServingRunner

APP_NAME = "flexsds-serving"
SERVING_CONFIG = json.loads(os.environ.get("FLEXSDS_SERVING_CONFIG", "{}"))
PORT = int(SERVING_CONFIG.get("port", 8000))
GPU = SERVING_CONFIG.get("target_gpu", "L4")
if GPU not in {"L4", "T4", "A10G"} or not 1 <= PORT <= 65535:
    raise ValueError("Unsupported Modal GPU or serving port")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("vllm==0.30.0", "fastapi", "psutil")
        .env({"PYTHONPATH": "/root/project:/root/project/src",
              "FLEXSDS_SERVING_CONFIG": json.dumps(SERVING_CONFIG)})
    )
    src_dir = PROJECT_ROOT / "src"
    if src_dir.exists():
        image = image.add_local_dir(str(src_dir), remote_path="/root/project/src")
    else:
        flexsds_dir = Path(__file__).resolve().parents[1]
        image = image.add_local_dir(str(flexsds_dir), remote_path="/root/project/src/flexsds")

    manifest_file = PROJECT_ROOT / "experiments/scalar_vllm_source_manifest.json"
    if manifest_file.exists():
        image = image.add_local_file(str(manifest_file),
                                     remote_path="/root/project/experiments/scalar_vllm_source_manifest.json")

    volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU,
                  volumes={"/root/.cache/huggingface": volume}, timeout=3600)
    @modal.web_server(PORT, startup_timeout=600)
    def serve():
        runner = LocalServingRunner()
        # Do not use the laptop's preflight here: the CLI evaluated the remote budget.
        # vLLM and Modal's web-server startup check remain the runtime authorities.
        subprocess.Popen(runner.build_command(SERVING_CONFIG),
                         env=runner.setup_environment(SERVING_CONFIG))
else:
    app = None
