"""[M5-01 web demo] Long-running OCR endpoint on Modal (Modal hosts models only).

Serves the pinned Qwen2.5-VL OCR model with exactly the server arguments of
``src/pipeline/ocr.py`` (``ocr_server_args``) behind Modal proxy
authentication. The web backend prepares images, builds requests and cleans
text with ``src/pipeline/ocr.py``; this app only runs ``vllm serve``.
The batch job ``experiments/modal_ocr.py`` stays unchanged.

Deploy from the repository root with the conda env ``doan-d22``:

    modal deploy -m experiments.modal_ocr_serve

Endpoint: ``ocr.ocr_endpoint_url(os.environ)``
(``https://<workspace>--ptit-ocr-serve-ocr.us-east.modal.direct/v1``).
Switches: PTIT_OCR_GPU (default A10G), PTIT_SCALEDOWN_SECONDS (default 300),
PTIT_KEEP_WARM=1 keeps one container up (costs credit).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = "/root/project"
HF_HOME = "/root/.cache/huggingface"
VLLM_VERSION = "0.30.0"
APP_NAME = "ptit-ocr-serve"
GPU = os.environ.get("PTIT_OCR_GPU", "A10G")
PORT = 8000
SCALEDOWN_SECONDS = int(os.environ.get("PTIT_SCALEDOWN_SECONDS", "300"))
KEEP_WARM = os.environ.get("PTIT_KEEP_WARM") == "1"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HOME": HF_HOME})
    .add_local_dir(REPO_ROOT / "src", f"{REMOTE_PROJECT_ROOT}/src", ignore=["**/__pycache__/**"])
)
# Same weights cache as the batch OCR job.
cache = modal.Volume.from_name("ptit-ocr-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.server(
    name="ocr",
    gpu=GPU,
    volumes={HF_HOME: cache},
    port=PORT,
    startup_timeout=25 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
    min_containers=1 if KEEP_WARM else 0,
    max_containers=1,
    target_concurrency=8,
)
class OcrServer:
    @modal.enter()
    def start(self) -> None:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)
        from src.pipeline import ocr

        self.proc = subprocess.Popen(["vllm", "serve", *ocr.ocr_server_args(port=PORT)])

    @modal.exit()
    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        cache.commit()
